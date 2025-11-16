// Function: sub_DFF6F0
// Address: 0xdff6f0
//
__int64 __fastcall sub_DFF6F0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // [rsp-48h] [rbp-48h] BYREF
  __int64 v14; // [rsp-40h] [rbp-40h]
  _QWORD *v15; // [rsp-38h] [rbp-38h]
  _QWORD *v16; // [rsp-30h] [rbp-30h]

  if ( !a1 )
    return 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) == 0 )
  {
    if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) > 1 )
      goto LABEL_4;
    return 0;
  }
  if ( *(_DWORD *)(a1 - 24) <= 1u )
    return 0;
LABEL_4:
  v2 = *(_QWORD *)(a1 + 8);
  v3 = (_QWORD *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v2 & 4) != 0 )
    v3 = (_QWORD *)*v3;
  v4 = sub_BCCE00(v3, 0x40u);
  v5 = sub_AD64C0(v4, 0, 0);
  v6 = sub_B98A20(v5, 0);
  if ( sub_DFF670(a1) )
  {
    v10 = sub_AD64C0(v4, -1, 0);
    v13 = a1;
    v16 = sub_B98A20(v10, -1);
    v11 = *(_QWORD *)(a1 + 8);
    v14 = a1;
    v15 = v6;
    v12 = (__int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v11 & 4) != 0 )
      v12 = (__int64 *)*v12;
    return sub_B9C770(v12, &v13, (__int64 *)4, 0, 1);
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 8);
    v13 = a1;
    v14 = a1;
    v15 = v6;
    v8 = (__int64 *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v7 & 4) != 0 )
      v8 = (__int64 *)*v8;
    return sub_B9C770(v8, &v13, (__int64 *)3, 0, 1);
  }
}
