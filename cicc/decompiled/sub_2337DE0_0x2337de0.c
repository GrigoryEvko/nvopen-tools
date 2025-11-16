// Function: sub_2337DE0
// Address: 0x2337de0
//
__int64 __fastcall sub_2337DE0(char *a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 result; // rax
  char *v5; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h]

  v5 = a1;
  v6 = a2;
  result = sub_95CB50((const void **)&v5, a3, a4);
  if ( (_BYTE)result && v6 )
  {
    if ( *v5 == 60 )
    {
      if ( v5[v6 - 1] != 62 )
        return 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
