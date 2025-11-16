// Function: sub_133AAC0
// Address: 0x133aac0
//
__int64 __fastcall sub_133AAC0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // r14d
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v14; // rdi
  char *v15; // rbx
  __int64 v16; // rax
  unsigned int v17; // ebx
  unsigned int v18; // ebx
  unsigned int v19; // ecx
  __int64 v20; // rsi

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
  if ( a6 )
  {
    v10 = 22;
    if ( a7 != 8 )
      goto LABEL_12;
    sub_133A740(a1);
  }
  if ( a4 && a5 )
  {
    v11 = *a5;
    v12 = qword_4F96BA0;
    if ( *a5 == 8 )
    {
      v10 = 0;
      *a4 = *(_QWORD *)qword_4F96BA0;
    }
    else
    {
      if ( (unsigned __int64)*a5 > 8 )
        v11 = 8;
      if ( (unsigned int)v11 >= 8 )
      {
        v14 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *a4 = *(_QWORD *)qword_4F96BA0;
        *(_QWORD *)((char *)a4 + (unsigned int)v11 - 8) = *(_QWORD *)(v12 + (unsigned int)v11 - 8);
        v15 = (char *)a4 - v14;
        v16 = v12 - (_QWORD)v15;
        v17 = (v11 + (_DWORD)v15) & 0xFFFFFFF8;
        if ( v17 >= 8 )
        {
          v18 = v17 & 0xFFFFFFF8;
          v19 = 0;
          do
          {
            v20 = v19;
            v19 += 8;
            *(_QWORD *)(v14 + v20) = *(_QWORD *)(v16 + v20);
          }
          while ( v19 < v18 );
        }
      }
      else if ( (v11 & 4) != 0 )
      {
        *(_DWORD *)a4 = *(_DWORD *)qword_4F96BA0;
        *(_DWORD *)((char *)a4 + (unsigned int)v11 - 4) = *(_DWORD *)(v12 + (unsigned int)v11 - 4);
      }
      else if ( (_DWORD)v11 )
      {
        *(_BYTE *)a4 = *(_BYTE *)qword_4F96BA0;
        if ( (v11 & 2) != 0 )
          *(_WORD *)((char *)a4 + (unsigned int)v11 - 2) = *(_WORD *)(v12 + (unsigned int)v11 - 2);
      }
      *a5 = v11;
      v10 = 22;
    }
  }
  else
  {
    v10 = 0;
  }
LABEL_12:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v10;
}
