// Function: sub_B54A90
// Address: 0xb54a90
//
__int64 __fastcall sub_B54A90(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  __int64 v3; // r12

  v1 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v2 = sub_BD2C40(88, v1);
  v3 = v2;
  if ( v2 )
    sub_B4DAA0(v2, a1, v1);
  return v3;
}
