// Function: sub_132D2D0
// Address: 0x132d2d0
//
__int64 __fastcall sub_132D2D0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // ebx
  int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  _DWORD v16[13]; // [rsp+Ch] [rbp-34h]

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
    v12 = *(_DWORD *)(sub_1322320(*(_QWORD *)(a2 + 16))[10] + 600LL);
    v16[0] = v12;
    if ( a4 && a5 )
    {
      v13 = *a5;
      if ( *a5 == 4 )
      {
        *a4 = v12;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 4 )
          v13 = 4;
        if ( (_DWORD)v13 )
        {
          v14 = 0;
          do
          {
            v15 = v14++;
            *((_BYTE *)a4 + v15) = *((_BYTE *)v16 + v15);
          }
          while ( v14 < (unsigned int)v13 );
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
