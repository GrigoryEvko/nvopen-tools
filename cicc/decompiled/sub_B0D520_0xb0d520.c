// Function: sub_B0D520
// Address: 0xb0d520
//
__int64 __fastcall sub_B0D520(__int64 a1)
{
  __int64 *v3; // rdi
  __int64 v4; // [rsp+10h] [rbp-40h]
  __int64 *v5[2]; // [rsp+20h] [rbp-30h] BYREF
  char v6; // [rsp+30h] [rbp-20h]

  if ( !a1 )
    return v4;
  sub_AF4640((__int64)v5, a1);
  if ( !v6 )
    return v4;
  v3 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v3 = (__int64 *)*v3;
  return sub_B0D000(v3, v5[0], (__int64)v5[1], 0, 1);
}
