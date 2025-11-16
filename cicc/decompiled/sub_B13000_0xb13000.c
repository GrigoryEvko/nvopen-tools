// Function: sub_B13000
// Address: 0xb13000
//
__int64 __fastcall sub_B13000(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // rax
  __int64 v4; // [rsp+20h] [rbp-30h] BYREF
  char v5; // [rsp+30h] [rbp-20h]

  v1 = sub_B11F60(a1 + 80);
  sub_AF47B0((__int64)&v4, *(unsigned __int64 **)(v1 + 16), *(unsigned __int64 **)(v1 + 24));
  if ( v5 )
    return v4;
  v3 = sub_B12000(a1 + 72);
  return sub_AF3FE0(v3);
}
