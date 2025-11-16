// Function: sub_133C2B0
// Address: 0x133c2b0
//
__int64 __fastcall sub_133C2B0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // ebx
  __int64 v12; // r13
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdi
  char *v17; // r12
  char *v18; // rcx
  unsigned int v19; // r12d
  unsigned int v20; // r12d
  unsigned int v21; // eax
  __int64 v22; // rsi
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

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
    v12 = *(_QWORD *)(a2 + 16);
    v13 = (_BYTE *)(__readfsqword(0) - 2664);
    if ( __readfsbyte(0xFFFFF8C8) )
      v13 = (_BYTE *)sub_1313D30((__int64)v13, 0);
    v14 = sub_1322110(v13, v12, 1, 0)[5];
    v23[0] = v14;
    if ( a4 && a5 )
    {
      v15 = *a5;
      if ( *a5 == 8 )
      {
        *a4 = v14;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 8 )
          v15 = 8;
        if ( (unsigned int)v15 >= 8 )
        {
          *a4 = v14;
          v16 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((char *)a4 + (unsigned int)v15 - 8) = *(_QWORD *)((char *)&v23[-1] + (unsigned int)v15);
          v17 = (char *)a4 - v16;
          v18 = (char *)((char *)v23 - v17);
          v19 = (v15 + (_DWORD)v17) & 0xFFFFFFF8;
          if ( v19 >= 8 )
          {
            v20 = v19 & 0xFFFFFFF8;
            v21 = 0;
            do
            {
              v22 = v21;
              v21 += 8;
              *(_QWORD *)(v16 + v22) = *(_QWORD *)&v18[v22];
            }
            while ( v21 < v20 );
          }
        }
        else if ( (v15 & 4) != 0 )
        {
          *(_DWORD *)a4 = v23[0];
          *(_DWORD *)((char *)a4 + (unsigned int)v15 - 4) = *(_DWORD *)((char *)v23 + (unsigned int)v15 - 4);
        }
        else if ( (_DWORD)v15 )
        {
          *(_BYTE *)a4 = v23[0];
          if ( (v15 & 2) != 0 )
            *(_WORD *)((char *)a4 + (unsigned int)v15 - 2) = *(_WORD *)((char *)v23 + (unsigned int)v15 - 2);
        }
        *a5 = v15;
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
