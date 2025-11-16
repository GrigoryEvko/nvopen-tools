// Function: sub_1327C60
// Address: 0x1327c60
//
__int64 __fastcall sub_1327C60(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  char *v15; // r12
  char *v16; // rcx
  unsigned int v17; // r12d
  unsigned int v18; // r12d
  unsigned int v19; // eax
  __int64 v20; // rsi
  _QWORD v21[7]; // [rsp+8h] [rbp-38h] BYREF

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
    v12 = sub_130B0E0(qword_4F96BA8 + 408);
    v21[0] = v12;
    if ( a4 && a5 )
    {
      v13 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = v12;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v13 = 8;
        if ( (unsigned int)v13 >= 8 )
        {
          *a4 = v12;
          v14 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(__int64 *)((char *)a4 + (unsigned int)v13 - 8) = *(_QWORD *)((char *)&v21[-1] + (unsigned int)v13);
          v15 = (char *)a4 - v14;
          v16 = (char *)((char *)v21 - v15);
          v17 = (v13 + (_DWORD)v15) & 0xFFFFFFF8;
          if ( v17 >= 8 )
          {
            v18 = v17 & 0xFFFFFFF8;
            v19 = 0;
            do
            {
              v20 = v19;
              v19 += 8;
              *(_QWORD *)(v14 + v20) = *(_QWORD *)&v16[v20];
            }
            while ( v19 < v18 );
          }
        }
        else if ( (v13 & 4) != 0 )
        {
          *(_DWORD *)a4 = v21[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v13 - 4) = *(_DWORD *)((char *)v21 + (unsigned int)v13 - 4);
        }
        else if ( (_DWORD)v13 )
        {
          *(_BYTE *)a4 = v21[0];
          if ( (v13 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v13 - 2) = *(_WORD *)((char *)v21 + (unsigned int)v13 - 2);
        }
        *a5 = v13;
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
