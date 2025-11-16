// Function: sub_2E791F0
// Address: 0x2e791f0
//
bool __fastcall sub_2E791F0(__int64 *a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // [rsp-38h] [rbp-38h] BYREF
  int v6; // [rsp-30h] [rbp-30h]
  __int64 v7; // [rsp-28h] [rbp-28h] BYREF
  int v8; // [rsp-20h] [rbp-20h]

  if ( (*(_BYTE *)(a1[1] + 904) & 0x10) != 0 )
    return 1;
  v2 = *a1;
  if ( (unsigned int)sub_A746B0((_QWORD *)(*a1 + 120))
    || !(unsigned __int8)sub_B2D610(v2, 41)
    || (*(_BYTE *)(v2 + 2) & 8) != 0 )
  {
    return 1;
  }
  v3 = sub_BA8DC0(*(_QWORD *)(*a1 + 40), (__int64)"llvm.dbg.cu", 11);
  v4 = 0;
  if ( v3 )
    v4 = sub_B91A00(v3);
  v8 = v4;
  v7 = v3;
  sub_BA95A0((__int64)&v7);
  v5 = v3;
  v6 = 0;
  sub_BA95A0((__int64)&v5);
  return v6 != v8;
}
