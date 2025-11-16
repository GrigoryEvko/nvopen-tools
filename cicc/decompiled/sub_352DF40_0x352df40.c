// Function: sub_352DF40
// Address: 0x352df40
//
__int64 __fastcall sub_352DF40(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // r12d
  _QWORD v10[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v11)(); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v12)(); // [rsp+18h] [rbp-18h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208C0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208C0);
  v7 = a2[4];
  v10[0] = v6 + 176;
  v12 = sub_352F960;
  v11 = sub_352E040;
  v8 = sub_29C2F90(a2, v7, (__int64)(a2 + 3), "ModuleDebugify: ", 0x10u, (__int64)v10);
  if ( v11 )
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v11)(v10, v10, 3);
  return v8;
}
