// Function: sub_3189B90
// Address: 0x3189b90
//
_QWORD *__fastcall sub_3189B90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  _QWORD *v4; // r12
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_22077B0(0x28u);
  v3 = v2;
  if ( v2 )
  {
    sub_318EB10(v2, 35, a2, a1);
    *(_DWORD *)(v3 + 32) = 11;
    *(_QWORD *)v3 = &unk_4A33AD0;
  }
  v6[0] = v3;
  v4 = sub_3189570(a1, (__int64)v6);
  if ( v6[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  return v4;
}
