// Function: sub_2CC8500
// Address: 0x2cc8500
//
__int64 __fastcall sub_2CC8500(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r12d
  _QWORD v15[11]; // [rsp+0h] [rbp-130h] BYREF
  char *v16; // [rsp+58h] [rbp-D8h]
  __int64 v17; // [rsp+60h] [rbp-D0h]
  int v18; // [rsp+68h] [rbp-C8h]
  char v19; // [rsp+6Ch] [rbp-C4h]
  char v20; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+90h] [rbp-A0h]
  __int64 v22; // [rsp+98h] [rbp-98h]
  __int64 v23; // [rsp+A0h] [rbp-90h]
  __int64 v24; // [rsp+A8h] [rbp-88h]
  __int64 v25; // [rsp+B0h] [rbp-80h]
  __int64 v26; // [rsp+B8h] [rbp-78h]
  __int64 v27; // [rsp+C0h] [rbp-70h]
  __int64 v28; // [rsp+C8h] [rbp-68h]
  _BYTE *v29; // [rsp+D0h] [rbp-60h]
  __int64 v30; // [rsp+D8h] [rbp-58h]
  _BYTE v31[32]; // [rsp+E0h] [rbp-50h] BYREF
  __int16 v32; // [rsp+100h] [rbp-30h]

  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F8144C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_20;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F8144C);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7 + 176;
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F875EC )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_19;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F875EC);
  v15[1] = a3;
  v29 = v31;
  v15[2] = v12 + 176;
  v16 = &v20;
  v30 = 0x400000000LL;
  v15[3] = v9;
  memset(&v15[4], 0, 56);
  v17 = 4;
  v18 = 0;
  v19 = 1;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v32 = 0;
  v15[0] = a2;
  v13 = sub_D4B3D0(a2);
  if ( (_BYTE)v13 )
    v13 = sub_2CC5900(v15);
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  if ( !v19 )
    _libc_free((unsigned __int64)v16);
  return v13;
}
