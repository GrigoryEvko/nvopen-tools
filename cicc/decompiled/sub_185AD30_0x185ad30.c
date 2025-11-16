// Function: sub_185AD30
// Address: 0x185ad30
//
bool __fastcall sub_185AD30(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // rsi
  int v8; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  sub_16AF710(&v8, dword_4FAB2E0, 0x64u);
  v3 = sub_1368AA0(a2, *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40));
  v5 = v3;
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 56LL) + 80LL);
  if ( v6 )
    v6 -= 24;
  v9[0] = sub_1368AA0(a2, v6);
  return sub_16AF500(v9, v8) > v5;
}
