// Function: sub_193D800
// Address: 0x193d800
//
__int64 __fastcall sub_193D800(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  unsigned int v18; // r12d
  _BYTE v20[16]; // [rsp+0h] [rbp-130h] BYREF
  __int64 (__fastcall *v21)(_QWORD *, __int64, int); // [rsp+10h] [rbp-120h]
  __int64 (*v22)(); // [rsp+18h] [rbp-118h]
  __int64 v23; // [rsp+20h] [rbp-110h]
  __int64 v24; // [rsp+28h] [rbp-108h]
  __int64 v25; // [rsp+30h] [rbp-100h]
  __int64 v26; // [rsp+38h] [rbp-F8h]
  _QWORD v27[2]; // [rsp+40h] [rbp-F0h] BYREF
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-E0h]
  __int64 (*v29)(); // [rsp+58h] [rbp-D8h]
  _BYTE *v30; // [rsp+60h] [rbp-D0h]
  __int64 v31; // [rsp+68h] [rbp-C8h]
  _BYTE v32[128]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+F0h] [rbp-40h]
  __int64 v34; // [rsp+F8h] [rbp-38h]
  __int64 v35; // [rsp+100h] [rbp-30h]
  __int64 v36; // [rsp+108h] [rbp-28h]

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9E06C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_27;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9E06C);
  v5 = *(__int64 **)(a1 + 8);
  v6 = v4;
  v7 = v4 + 160;
  v8 = *v5;
  v9 = v5[1];
  if ( v8 == v9 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F9920C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_25;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F9920C);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10 + 160;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F99CCC )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_26;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F99CCC);
  v16 = *(_QWORD *)(v6 + 216);
  v25 = v12;
  v23 = v7;
  v22 = sub_19395A0;
  v26 = v16;
  v21 = sub_1939150;
  v24 = v15 + 160;
  v28 = 0;
  sub_1939150(v27, (__int64)v20, 2);
  v30 = v32;
  v33 = 0;
  v29 = v22;
  v34 = 0;
  v28 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v21;
  v31 = 0x1000000000LL;
  v35 = 0;
  v36 = 0;
  v18 = sub_193C710(v17);
  j___libc_free_0(v34);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  if ( v28 )
    v28(v27, v27, 3);
  if ( v21 )
    v21(v20, (__int64)v20, 3);
  return v18;
}
