// Function: sub_270DB60
// Address: 0x270db60
//
__int64 __fastcall sub_270DB60(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v19; // rax
  unsigned __int8 v20[32]; // [rsp+0h] [rbp-150h] BYREF
  __int64 v21; // [rsp+20h] [rbp-130h]
  __int64 v22; // [rsp+28h] [rbp-128h]
  __int64 v23; // [rsp+30h] [rbp-120h]
  unsigned int v24; // [rsp+38h] [rbp-118h]
  __int64 v25; // [rsp+40h] [rbp-110h]
  _QWORD *v26; // [rsp+48h] [rbp-108h]
  __int64 v27; // [rsp+50h] [rbp-100h]
  unsigned int v28; // [rsp+58h] [rbp-F8h]
  __int64 v29; // [rsp+60h] [rbp-F0h]
  __int64 v30; // [rsp+68h] [rbp-E8h]
  __int64 v31; // [rsp+70h] [rbp-E0h]
  __int64 v32; // [rsp+78h] [rbp-D8h]
  __int64 v33; // [rsp+80h] [rbp-D0h]
  __int64 v34; // [rsp+88h] [rbp-C8h]
  __int64 v35; // [rsp+90h] [rbp-C0h]
  __int64 v36; // [rsp+98h] [rbp-B8h]
  __int64 v37; // [rsp+A0h] [rbp-B0h]
  __int64 v38; // [rsp+A8h] [rbp-A8h]
  __int64 v39; // [rsp+B0h] [rbp-A0h]
  __int64 v40; // [rsp+B8h] [rbp-98h]
  bool v41; // [rsp+C0h] [rbp-90h]
  _BYTE *v42; // [rsp+C8h] [rbp-88h]
  __int64 v43; // [rsp+D0h] [rbp-80h]
  char *v44; // [rsp+D8h] [rbp-78h]
  __int64 v45; // [rsp+E0h] [rbp-70h]
  int v46; // [rsp+E8h] [rbp-68h]
  char v47; // [rsp+ECh] [rbp-64h]
  char v48; // [rsp+F0h] [rbp-60h] BYREF

  v44 = &v48;
  v20[1] = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v43 = 0;
  v45 = 8;
  v46 = 0;
  v47 = 1;
  v2 = *(_QWORD *)(a2 + 40);
  v41 = sub_270A460(v2);
  if ( v41 )
  {
    v29 = v2;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v19 = (_BYTE *)sub_BA91D0(v2, "clang.arc.retainAutoreleasedReturnValueMarker", 0x2Du);
    if ( v19 && *v19 )
      v19 = 0;
    v42 = v19;
  }
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F86530 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_36;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F86530);
  v7 = *(__int64 **)(a1 + 8);
  v8 = *(_QWORD *)(v6 + 176);
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F8144C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_35;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8144C);
  v12 = v41;
  if ( v41 )
  {
    v12 = byte_5031DC8[0];
    if ( byte_5031DC8[0] )
      v12 = sub_270B770(v20, a2, v8, v11 + 176);
  }
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
  v13 = v28;
  if ( v28 )
  {
    v14 = v26;
    v15 = &v26[7 * v28];
    do
    {
      if ( *v14 != -4096 && *v14 != -8192 )
      {
        v16 = v14[6];
        if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
          sub_BD60C0(v14 + 4);
        v17 = v14[3];
        if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
          sub_BD60C0(v14 + 1);
      }
      v14 += 7;
    }
    while ( v15 != v14 );
    v13 = v28;
  }
  sub_C7D6A0((__int64)v26, 56 * v13, 8);
  sub_C7D6A0(v22, 24LL * v24, 8);
  return v12;
}
