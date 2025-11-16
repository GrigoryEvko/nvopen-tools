// Function: sub_18DBD70
// Address: 0x18dbd70
//
__int64 __fastcall sub_18DBD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v5; // bl
  __int64 result; // rax
  unsigned __int8 v7; // [rsp+Fh] [rbp-11h]

  v5 = *(_BYTE *)(a1 + 2);
  result = sub_18DC2F0(a2, a3, a4, a5);
  if ( (_BYTE)result )
  {
    if ( v5 == 3 )
    {
      v7 = result;
      sub_18DB890(a1, 2);
      return v7;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
