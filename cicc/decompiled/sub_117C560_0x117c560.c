// Function: sub_117C560
// Address: 0x117c560
//
unsigned __int8 *__fastcall sub_117C560(const __m128i *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int8 v5; // al
  unsigned __int8 v6; // cl
  unsigned __int8 *result; // rax
  __int64 v8; // rax
  __int64 v9; // [rsp+0h] [rbp-30h]
  const __m128i *v10[2]; // [rsp+18h] [rbp-18h] BYREF

  v5 = *(_BYTE *)a3;
  v6 = *(_BYTE *)a4;
  v10[0] = a1;
  if ( (unsigned __int8)(v5 - 42) <= 0x11u )
  {
    v8 = *(_QWORD *)(a3 + 16);
    if ( v8 )
    {
      if ( !*(_QWORD *)(v8 + 8) && v6 > 0x15u )
      {
        v9 = a3;
        result = sub_117B470(v10, a2, (unsigned __int8 *)a3, a4, 0);
        a3 = v9;
        if ( result )
          return result;
        v6 = *(_BYTE *)a4;
      }
    }
  }
  result = 0;
  if ( (unsigned __int8)(v6 - 42) <= 0x11u )
  {
    result = (unsigned __int8 *)a4[2];
    if ( result )
    {
      result = (unsigned __int8 *)*((_QWORD *)result + 1);
      if ( result )
      {
        return 0;
      }
      else if ( *(_BYTE *)a3 > 0x15u )
      {
        return sub_117B470(v10, a2, (unsigned __int8 *)a4, (__int64 *)a3, 1);
      }
    }
  }
  return result;
}
