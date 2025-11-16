// Function: sub_70C700
// Address: 0x70c700
//
__int64 __fastcall sub_70C700(unsigned __int8 a1, __m128i *a2, __m128i *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v7; // rsi
  int v8; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v9; // [rsp+4h] [rbp-4Ch] BYREF
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  __m128i v11[4]; // [rsp+10h] [rbp-40h] BYREF

  *a4 = 0;
  v8 = 0;
  result = sub_70B8A0(a1, a2);
  if ( (_DWORD)result || (result = sub_709C40(a2, a1), (_DWORD)result) || (result = sub_709CC0(a2, a1), (_DWORD)result) )
  {
    *a3 = _mm_loadu_si128(a2);
  }
  else
  {
    sub_70B720(a1, a2, &v10, a4, &v8);
    if ( v8 )
      *a4 = 1;
    sub_70B680(a1, v10, v11, a4);
    if ( (unsigned int)sub_70BE30(a1, a2, v11, &v9) )
    {
      if ( (unsigned int)sub_70C5B0(a1, (unsigned int *)a2) )
      {
        result = sub_70B680(a1, v10, a3, a4);
        if ( !v10 )
          return sub_70BAF0(a1, a3, a3, a4, a4);
      }
      else
      {
        v7 = v10;
        if ( v10 == 0x7FFFFFFFFFFFFFFFLL )
          *a4 = 1;
        else
          v7 = ++v10;
        return sub_70B680(a1, v7, a3, a4);
      }
    }
    else
    {
      result = v9;
      *a3 = _mm_loadu_si128(a2);
      if ( (_DWORD)result )
        *a4 = 1;
    }
  }
  return result;
}
