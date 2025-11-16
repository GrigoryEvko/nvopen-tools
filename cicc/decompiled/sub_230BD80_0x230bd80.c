// Function: sub_230BD80
// Address: 0x230bd80
//
_QWORD *__fastcall sub_230BD80(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  int v9; // ecx
  unsigned __int64 v10; // rax
  _QWORD *v11; // rdx
  int v12; // ecx
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  int v15; // edx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rdi
  char v23[8]; // [rsp+0h] [rbp-150h] BYREF
  int v24; // [rsp+8h] [rbp-148h] BYREF
  _QWORD *v25; // [rsp+10h] [rbp-140h]
  int *v26; // [rsp+18h] [rbp-138h]
  int *v27; // [rsp+20h] [rbp-130h]
  __int64 v28; // [rsp+28h] [rbp-128h]
  int v29; // [rsp+38h] [rbp-118h] BYREF
  unsigned __int64 v30; // [rsp+40h] [rbp-110h]
  int *v31; // [rsp+48h] [rbp-108h]
  int *v32; // [rsp+50h] [rbp-100h]
  __int64 v33; // [rsp+58h] [rbp-F8h]
  int v34; // [rsp+68h] [rbp-E8h] BYREF
  _QWORD *v35; // [rsp+70h] [rbp-E0h]
  int *v36; // [rsp+78h] [rbp-D8h]
  int *v37; // [rsp+80h] [rbp-D0h]
  __int64 v38; // [rsp+88h] [rbp-C8h]
  int v39; // [rsp+98h] [rbp-B8h] BYREF
  _QWORD *v40; // [rsp+A0h] [rbp-B0h]
  int *v41; // [rsp+A8h] [rbp-A8h]
  int *v42; // [rsp+B0h] [rbp-A0h]
  __int64 v43; // [rsp+B8h] [rbp-98h]
  int v44; // [rsp+C8h] [rbp-88h] BYREF
  unsigned __int64 v45; // [rsp+D0h] [rbp-80h]
  int *v46; // [rsp+D8h] [rbp-78h]
  int *v47; // [rsp+E0h] [rbp-70h]
  __int64 v48; // [rsp+E8h] [rbp-68h]
  int v49; // [rsp+F8h] [rbp-58h] BYREF
  _QWORD *v50; // [rsp+100h] [rbp-50h]
  int *v51; // [rsp+108h] [rbp-48h]
  int *v52; // [rsp+110h] [rbp-40h]
  __int64 v53; // [rsp+118h] [rbp-38h]

  sub_30A9120(v23, a2 + 8);
  if ( v25 )
  {
    v40 = v25;
    v39 = v24;
    v41 = v26;
    v42 = v27;
    v25[1] = &v39;
    v25 = 0;
    v43 = v28;
    v26 = &v24;
    v27 = &v24;
    v3 = v30;
    v28 = 0;
    if ( v30 )
      goto LABEL_3;
LABEL_22:
    v4 = v35;
    v44 = 0;
    v45 = 0;
    v46 = &v44;
    v47 = &v44;
    v48 = 0;
    if ( v35 )
      goto LABEL_4;
    goto LABEL_23;
  }
  v3 = v30;
  v39 = 0;
  v40 = 0;
  v41 = &v39;
  v42 = &v39;
  v43 = 0;
  if ( !v30 )
    goto LABEL_22;
LABEL_3:
  v45 = v3;
  v44 = v29;
  v46 = v31;
  v47 = v32;
  *(_QWORD *)(v3 + 8) = &v44;
  v30 = 0;
  v48 = v33;
  v31 = &v29;
  v32 = &v29;
  v4 = v35;
  v33 = 0;
  if ( v35 )
  {
LABEL_4:
    v50 = v4;
    v49 = v34;
    v51 = v36;
    v52 = v37;
    v4[1] = &v49;
    v35 = 0;
    v53 = v38;
    v36 = &v34;
    v37 = &v34;
    v38 = 0;
    goto LABEL_5;
  }
LABEL_23:
  v49 = 0;
  v50 = 0;
  v51 = &v49;
  v52 = &v49;
  v53 = 0;
LABEL_5:
  v5 = (_QWORD *)sub_22077B0(0x98u);
  v6 = v5;
  if ( v5 )
  {
    v7 = v5 + 2;
    *v5 = &unk_4A0B470;
    v8 = v40;
    if ( v40 )
    {
      v9 = v39;
      v6[3] = v40;
      *((_DWORD *)v6 + 4) = v9;
      v6[4] = v41;
      v6[5] = v42;
      v8[1] = v7;
      v40 = 0;
      v6[6] = v43;
      v41 = &v39;
      v42 = &v39;
      v43 = 0;
    }
    else
    {
      *((_DWORD *)v6 + 4) = 0;
      v6[3] = 0;
      v6[4] = v7;
      v6[5] = v7;
      v6[6] = 0;
    }
    v10 = v45;
    v11 = v6 + 8;
    if ( v45 )
    {
      v12 = v44;
      v6[9] = v45;
      *((_DWORD *)v6 + 16) = v12;
      v6[10] = v46;
      v6[11] = v47;
      *(_QWORD *)(v10 + 8) = v11;
      v45 = 0;
      v6[12] = v48;
      v46 = &v44;
      v47 = &v44;
      v48 = 0;
    }
    else
    {
      *((_DWORD *)v6 + 16) = 0;
      v6[9] = 0;
      v6[10] = v11;
      v6[11] = v11;
      v6[12] = 0;
    }
    v13 = v50;
    v14 = v6 + 14;
    if ( v50 )
    {
      v15 = v49;
      v6[15] = v50;
      *((_DWORD *)v6 + 28) = v15;
      v6[16] = v51;
      v6[17] = v52;
      v13[1] = v14;
      v13 = 0;
      v50 = 0;
      v6[18] = v53;
      v51 = &v49;
      v52 = &v49;
      v53 = 0;
    }
    else
    {
      *((_DWORD *)v6 + 28) = 0;
      v6[15] = 0;
      v6[16] = v14;
      v6[17] = v14;
      v6[18] = 0;
    }
  }
  else
  {
    v13 = v50;
  }
  sub_230BB90(v13);
  v16 = v45;
  while ( v16 )
  {
    v17 = v16;
    sub_23092F0(*(_QWORD **)(v16 + 24));
    v18 = *(_QWORD *)(v16 + 40);
    v16 = *(_QWORD *)(v16 + 16);
    if ( v18 != v17 + 56 )
      _libc_free(v18);
    j_j___libc_free_0(v17);
  }
  sub_230BAD0(v40);
  *a1 = v6;
  sub_230BB90(v35);
  v19 = v30;
  while ( v19 )
  {
    v20 = v19;
    sub_23092F0(*(_QWORD **)(v19 + 24));
    v21 = *(_QWORD *)(v19 + 40);
    v19 = *(_QWORD *)(v19 + 16);
    if ( v21 != v20 + 56 )
      _libc_free(v21);
    j_j___libc_free_0(v20);
  }
  sub_230BAD0(v25);
  return a1;
}
