// Function: sub_1335860
// Address: 0x1335860
//
__int64 __fastcall sub_1335860(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // ebx
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // r12
  char *v17; // rcx
  unsigned int v18; // r12d
  unsigned int v19; // r12d
  unsigned int v20; // eax
  __int64 v21; // rsi
  _QWORD v22[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v10 = 1;
  if ( !(a7 | a6) )
  {
    v12 = sub_1322320(*(_QWORD *)(a2 + 16));
    v13 = sub_130B0E0(v12[10] + 376LL);
    v22[0] = v13;
    if ( a4 && a5 )
    {
      v14 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = v13;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v14 = 8;
        if ( (unsigned int)v14 >= 8 )
        {
          *a4 = v13;
          v15 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(__int64 *)((char *)a4 + (unsigned int)v14 - 8) = *(_QWORD *)((char *)&v22[-1] + (unsigned int)v14);
          v16 = (char *)a4 - v15;
          v17 = (char *)((char *)v22 - v16);
          v18 = (v14 + (_DWORD)v16) & 0xFFFFFFF8;
          if ( v18 >= 8 )
          {
            v19 = v18 & 0xFFFFFFF8;
            v20 = 0;
            do
            {
              v21 = v20;
              v20 += 8;
              *(_QWORD *)(v15 + v21) = *(_QWORD *)&v17[v21];
            }
            while ( v20 < v19 );
          }
        }
        else if ( (v14 & 4) != 0 )
        {
          *(_DWORD *)a4 = v22[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v14 - 4) = *(_DWORD *)((char *)v22 + (unsigned int)v14 - 4);
        }
        else if ( (_DWORD)v14 )
        {
          *(_BYTE *)a4 = v22[0];
          if ( (v14 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v14 - 2) = *(_WORD *)((char *)v22 + (unsigned int)v14 - 2);
        }
        *a5 = v14;
        v10 = 22;
      }
    }
    else
    {
      v10 = 0;
    }
  }
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v10;
}
