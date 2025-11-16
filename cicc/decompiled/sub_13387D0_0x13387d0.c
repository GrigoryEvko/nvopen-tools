// Function: sub_13387D0
// Address: 0x13387d0
//
__int64 __fastcall sub_13387D0(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        _DWORD *a4,
        __int64 *a5,
        const __m128i *a6,
        __int64 a7)
{
  unsigned int v10; // r15d
  int v11; // eax
  __int64 v12; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  int v16; // [rsp+Ch] [rbp-44h]
  __m128i v17[4]; // [rsp+10h] [rbp-40h] BYREF

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_BYTE *)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v17[0] = _mm_loadu_si128((const __m128i *)&off_49E8000);
  if ( !a4 || !a5 || *a5 != 4 )
  {
    *a5 = 0;
    v10 = 22;
    goto LABEL_14;
  }
  if ( a6 )
  {
    v10 = 22;
    if ( a7 != 16 )
      goto LABEL_14;
    v17[0] = _mm_loadu_si128(a6);
  }
  v11 = sub_1322200(a1, (__int64)v17);
  v16 = v11;
  if ( v11 == -1 )
  {
    v10 = 11;
  }
  else
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
          *((_BYTE *)a4 + v15) = *((_BYTE *)&v16 + v15);
        }
        while ( v14 < (unsigned int)v12 );
      }
      *a5 = v12;
      v10 = 22;
    }
  }
LABEL_14:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v10;
}
