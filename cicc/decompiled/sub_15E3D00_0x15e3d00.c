// Function: sub_15E3D00
// Address: 0x15e3d00
//
__int64 __fastcall sub_15E3D00(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int64 *v4; // rcx
  unsigned __int64 v5; // rdx

  v1 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 40) = 0;
  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
  {
    v2 = *(_QWORD *)(v1 + 120);
    if ( v2 )
    {
      v3 = sub_16498B0(a1);
      sub_164D860(v2, v3);
    }
  }
  v4 = *(unsigned __int64 **)(a1 + 64);
  v5 = *(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL;
  *v4 = v5 | *v4 & 7;
  *(_QWORD *)(v5 + 8) = v4;
  *(_QWORD *)(a1 + 56) &= 7uLL;
  *(_QWORD *)(a1 + 64) = 0;
  sub_15E3C20((_QWORD *)a1);
  return sub_1648B90(a1);
}
