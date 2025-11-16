// Function: sub_270D670
// Address: 0x270d670
//
__int64 __fastcall sub_270D670(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v15; // rax
  __int64 v16; // [rsp+0h] [rbp-1C0h] BYREF
  _QWORD *v17; // [rsp+8h] [rbp-1B8h]
  __int64 v18; // [rsp+10h] [rbp-1B0h]
  int v19; // [rsp+18h] [rbp-1A8h]
  char v20; // [rsp+1Ch] [rbp-1A4h]
  _QWORD v21[2]; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v22; // [rsp+30h] [rbp-190h] BYREF
  _BYTE *v23; // [rsp+38h] [rbp-188h]
  __int64 v24; // [rsp+40h] [rbp-180h]
  int v25; // [rsp+48h] [rbp-178h]
  char v26; // [rsp+4Ch] [rbp-174h]
  _BYTE v27[16]; // [rsp+50h] [rbp-170h] BYREF
  unsigned __int8 v28; // [rsp+60h] [rbp-160h] BYREF
  char v29; // [rsp+61h] [rbp-15Fh]
  __int64 v30; // [rsp+80h] [rbp-140h]
  __int64 v31; // [rsp+88h] [rbp-138h]
  __int64 v32; // [rsp+90h] [rbp-130h]
  unsigned int v33; // [rsp+98h] [rbp-128h]
  __int64 v34; // [rsp+A0h] [rbp-120h]
  _QWORD *v35; // [rsp+A8h] [rbp-118h]
  __int64 v36; // [rsp+B0h] [rbp-110h]
  unsigned int v37; // [rsp+B8h] [rbp-108h]
  __int64 v38; // [rsp+C0h] [rbp-100h]
  __int64 v39; // [rsp+C8h] [rbp-F8h]
  __int64 v40; // [rsp+D0h] [rbp-F0h]
  __int64 v41; // [rsp+D8h] [rbp-E8h]
  __int64 v42; // [rsp+E0h] [rbp-E0h]
  __int64 v43; // [rsp+E8h] [rbp-D8h]
  __int64 v44; // [rsp+F0h] [rbp-D0h]
  __int64 v45; // [rsp+F8h] [rbp-C8h]
  __int64 v46; // [rsp+100h] [rbp-C0h]
  __int64 v47; // [rsp+108h] [rbp-B8h]
  __int64 v48; // [rsp+110h] [rbp-B0h]
  __int64 v49; // [rsp+118h] [rbp-A8h]
  bool v50; // [rsp+120h] [rbp-A0h]
  _BYTE *v51; // [rsp+128h] [rbp-98h]
  __int64 v52; // [rsp+130h] [rbp-90h]
  char *v53; // [rsp+138h] [rbp-88h]
  __int64 v54; // [rsp+140h] [rbp-80h]
  int v55; // [rsp+148h] [rbp-78h]
  char v56; // [rsp+14Ch] [rbp-74h]
  char v57; // [rsp+150h] [rbp-70h] BYREF

  v53 = &v57;
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
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v52 = 0;
  v54 = 8;
  v55 = 0;
  v56 = 1;
  v6 = *(_QWORD *)(a3 + 40);
  v50 = sub_270A460(v6);
  if ( v50 )
  {
    v38 = v6;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v15 = (_BYTE *)sub_BA91D0(v6, "clang.arc.retainAutoreleasedReturnValueMarker", 0x2Du);
    if ( v15 && *v15 )
      v15 = 0;
    v51 = v15;
  }
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = sub_BC1CD0(a4, &unk_4F86540, a3);
  if ( v50 && unk_5031DC8 && (unsigned __int8)sub_270B770(&v28, a3, v8 + 8, v7 + 8) )
  {
    v16 = 0;
    v17 = v21;
    v18 = 2;
    v19 = 0;
    v20 = 1;
    v22 = 0;
    v23 = v27;
    v24 = 2;
    v25 = 0;
    v26 = 1;
    if ( !v29 )
    {
      HIDWORD(v18) = 1;
      v16 = 1;
      v21[0] = &unk_4F82408;
    }
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v21, (__int64)&v16);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v27, (__int64)&v22);
    if ( !v26 )
      _libc_free((unsigned __int64)v23);
    if ( !v20 )
    {
      _libc_free((unsigned __int64)v17);
      if ( v56 )
        goto LABEL_6;
      goto LABEL_26;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  if ( v56 )
    goto LABEL_6;
LABEL_26:
  _libc_free((unsigned __int64)v53);
LABEL_6:
  v9 = v37;
  if ( v37 )
  {
    v10 = v35;
    v11 = &v35[7 * v37];
    do
    {
      if ( *v10 != -4096 && *v10 != -8192 )
      {
        v12 = v10[6];
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0(v10 + 4);
        v13 = v10[3];
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v10 + 1);
      }
      v10 += 7;
    }
    while ( v11 != v10 );
    v9 = v37;
  }
  sub_C7D6A0((__int64)v35, 56 * v9, 8);
  sub_C7D6A0(v31, 24LL * v33, 8);
  return a1;
}
