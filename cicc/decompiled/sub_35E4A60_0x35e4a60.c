// Function: sub_35E4A60
// Address: 0x35e4a60
//
__int64 __fastcall sub_35E4A60(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // r14d
  __int64 v17; // rax
  int v19; // [rsp+0h] [rbp-1D0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-1C8h]
  __int64 v21; // [rsp+10h] [rbp-1C0h]
  int v22; // [rsp+18h] [rbp-1B8h]
  __int64 v23; // [rsp+20h] [rbp-1B0h]
  char *v24; // [rsp+28h] [rbp-1A8h]
  __int64 v25; // [rsp+30h] [rbp-1A0h]
  int v26; // [rsp+38h] [rbp-198h]
  char v27; // [rsp+3Ch] [rbp-194h]
  char v28; // [rsp+40h] [rbp-190h] BYREF
  __int64 v29; // [rsp+C0h] [rbp-110h]
  char *v30; // [rsp+C8h] [rbp-108h]
  __int64 v31; // [rsp+D0h] [rbp-100h]
  int v32; // [rsp+D8h] [rbp-F8h]
  char v33; // [rsp+DCh] [rbp-F4h]
  char v34; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+120h] [rbp-B0h]
  char *v36; // [rsp+128h] [rbp-A8h]
  __int64 v37; // [rsp+130h] [rbp-A0h]
  int v38; // [rsp+138h] [rbp-98h]
  char v39; // [rsp+13Ch] [rbp-94h]
  char v40; // [rsp+140h] [rbp-90h] BYREF
  __int64 v41; // [rsp+160h] [rbp-70h]
  char *v42; // [rsp+168h] [rbp-68h]
  __int64 v43; // [rsp+170h] [rbp-60h]
  int v44; // [rsp+178h] [rbp-58h]
  char v45; // [rsp+17Ch] [rbp-54h]
  char v46; // [rsp+180h] [rbp-50h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_30;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5027190);
  v6 = *(__int64 **)(a1 + 8);
  v7 = *(_QWORD *)(v5 + 256);
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F89C28 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_28;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F89C28);
  v11 = sub_DFED00(v10, a2);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F875EC )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_29;
  }
  v16 = 0;
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F875EC);
  v45 = 1;
  v19 = 0;
  v24 = &v28;
  v30 = &v34;
  v36 = &v40;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v25 = 16;
  v26 = 0;
  v27 = 1;
  v29 = 0;
  v31 = 8;
  v32 = 0;
  v33 = 1;
  v35 = 0;
  v37 = 4;
  v38 = 0;
  v39 = 1;
  v41 = 0;
  v42 = &v46;
  v43 = 4;
  v44 = 0;
  if ( !(_BYTE)qword_5040288 )
  {
    v16 = sub_35E4090((__int64)&v19, a2, v7, v13, v17 + 176);
    if ( !v45 )
      _libc_free((unsigned __int64)v42);
    if ( !v39 )
      _libc_free((unsigned __int64)v36);
    if ( !v33 )
      _libc_free((unsigned __int64)v30);
  }
  if ( !v27 )
    _libc_free((unsigned __int64)v24);
  return v16;
}
