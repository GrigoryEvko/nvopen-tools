// Function: sub_30206C0
// Address: 0x30206c0
//
__int64 __fastcall sub_30206C0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  bool v5; // zf
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = sub_22077B0(0x4B0u);
  v4 = v3;
  if ( v3 )
  {
    v7[0] = v2;
    sub_31DA2B0(v3, a1, v7);
    if ( v7[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7[0] + 56LL))(v7[0]);
    *(_QWORD *)(v4 + 984) = 0;
    v5 = *(_DWORD *)(a1 + 1280) == 1;
    *(_QWORD *)v4 = off_4A2E3B0;
    *(_QWORD *)(v4 + 1000) = v4 + 1016;
    *(_QWORD *)(v4 + 1008) = 0x400000000LL;
    *(_QWORD *)(v4 + 1048) = v4 + 1064;
    *(_QWORD *)(v4 + 992) = 0;
    *(_QWORD *)(v4 + 1056) = 0;
    *(_QWORD *)(v4 + 1064) = 0;
    *(_QWORD *)(v4 + 1072) = 1;
    *(_QWORD *)(v4 + 1088) = 0;
    *(_QWORD *)(v4 + 1112) = 0;
    *(_QWORD *)(v4 + 1120) = 0;
    *(_QWORD *)(v4 + 1128) = 0;
    *(_DWORD *)(v4 + 1136) = 0;
    *(_DWORD *)(v4 + 1152) = 0;
    *(_QWORD *)(v4 + 1160) = 0;
    *(_QWORD *)(v4 + 1168) = v4 + 1152;
    *(_QWORD *)(v4 + 1176) = v4 + 1152;
    *(_QWORD *)(v4 + 1184) = 0;
    *(_BYTE *)(v4 + 1192) = v5;
    return v4;
  }
  if ( !v2 )
    return v4;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2);
  return 0;
}
