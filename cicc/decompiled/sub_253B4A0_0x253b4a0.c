// Function: sub_253B4A0
// Address: 0x253b4a0
//
_BOOL8 __fastcall sub_253B4A0(__int64 a1, __int64 a2)
{
  char v2; // r8
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v6; // [rsp+8h] [rbp-58h] BYREF
  __int128 v7; // [rsp+10h] [rbp-50h] BYREF
  __int128 v8; // [rsp+20h] [rbp-40h]
  _QWORD v9[5]; // [rsp+30h] [rbp-30h] BYREF

  v9[0] = &v6;
  v9[1] = a2;
  v6 = 0;
  v9[2] = a1;
  v9[3] = &v7;
  v7 = 0;
  v8 = 0;
  v2 = sub_2527330(a2, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2587130, (__int64)v9, a1, 1u, 1u);
  v3 = 1;
  if ( v2 )
  {
    v3 = 0x100000000LL;
    if ( BYTE8(v8) )
    {
      v3 = 1;
      if ( (_QWORD)v8 )
        v3 = v8;
      if ( v3 > 0x100000000LL )
        v3 = 0x100000000LL;
    }
  }
  v4 = *(_QWORD *)(a1 + 104);
  if ( v4 <= v3 )
    v3 = *(_QWORD *)(a1 + 104);
  if ( v3 < *(_QWORD *)(a1 + 96) )
    v3 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 104) = v3;
  return v4 == v3;
}
