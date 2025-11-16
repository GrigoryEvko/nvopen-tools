// Function: sub_13285C0
// Address: 0x13285c0
//
__int64 __fastcall sub_13285C0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // r15d
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  char *v14; // r12
  char *v15; // rcx
  unsigned int v16; // r12d
  unsigned int v17; // r12d
  unsigned int v18; // eax
  __int64 v19; // rsi
  _QWORD v20[7]; // [rsp+8h] [rbp-38h] BYREF

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
    v20[0] = qword_4F96998;
    if ( a4 && a5 )
    {
      v12 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = qword_4F96998;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v12 = 8;
        if ( (unsigned int)v12 >= 8 )
        {
          *a4 = qword_4F96998;
          v13 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((char *)a4 + (unsigned int)v12 - 8) = *(_QWORD *)((char *)&v20[-1] + (unsigned int)v12);
          v14 = (char *)a4 - v13;
          v15 = (char *)((char *)v20 - v14);
          v16 = (v12 + (_DWORD)v14) & 0xFFFFFFF8;
          if ( v16 >= 8 )
          {
            v17 = v16 & 0xFFFFFFF8;
            v18 = 0;
            do
            {
              v19 = v18;
              v18 += 8;
              *(_QWORD *)(v13 + v19) = *(_QWORD *)&v15[v19];
            }
            while ( v18 < v17 );
          }
        }
        else if ( (v12 & 4) != 0 )
        {
          *(_DWORD *)a4 = v20[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v12 - 4) = *(_DWORD *)((char *)v20 + (unsigned int)v12 - 4);
        }
        else if ( (_DWORD)v12 )
        {
          *(_BYTE *)a4 = v20[0];
          if ( (v12 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v12 - 2) = *(_WORD *)((char *)v20 + (unsigned int)v12 - 2);
        }
        *a5 = v12;
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
