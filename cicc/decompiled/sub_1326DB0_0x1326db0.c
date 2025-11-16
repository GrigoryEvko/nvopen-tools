// Function: sub_1326DB0
// Address: 0x1326db0
//
__int64 __fastcall sub_1326DB0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 *a5, __int64 a6, __int64 a7)
{
  unsigned int v10; // r15d
  int v11; // eax
  __int64 v12; // rdx
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
    v11 = *(_DWORD *)(qword_4F96BA8 + 432);
    v16[0] = v11;
    if ( a4 && a5 )
    {
      v12 = *a5;
      if ( *a5 == 4 )
      {
        *a4 = v11;
        v10 = 0;
      }
      else
      {
        if ( (unsigned __int64)*a5 > 4 )
          v12 = 4;
        if ( (_DWORD)v12 )
        {
          v14 = 0;
          do
          {
            v15 = v14++;
            *((_BYTE *)a4 + v15) = *((_BYTE *)v16 + v15);
          }
          while ( v14 < (unsigned int)v12 );
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
