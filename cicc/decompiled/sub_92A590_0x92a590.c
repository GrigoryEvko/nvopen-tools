// Function: sub_92A590
// Address: 0x92a590
//
__int64 __fastcall sub_92A590(_QWORD *a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v8; // rbx
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int **v12; // r11
  _BYTE *v13; // r15
  _BYTE *v14; // rax
  __int64 v15; // rax
  char j; // dl
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int **v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int **v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int **v27; // rdi
  __int64 v28; // rcx
  unsigned int **v30; // rdi
  unsigned int v31; // r14d
  unsigned int v32; // eax
  __int64 v33; // rax
  _BYTE *v34; // r10
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 (__fastcall *v37)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned int **v40; // r11
  unsigned int *v41; // rax
  __int64 v42; // rax
  unsigned int **v43; // [rsp+10h] [rbp-C0h]
  unsigned int **v44; // [rsp+10h] [rbp-C0h]
  unsigned int **v45; // [rsp+10h] [rbp-C0h]
  _BYTE *v46; // [rsp+18h] [rbp-B8h]
  __int64 v47; // [rsp+18h] [rbp-B8h]
  __int64 v48; // [rsp+20h] [rbp-B0h]
  unsigned int **v49; // [rsp+28h] [rbp-A8h]
  __int64 v50; // [rsp+28h] [rbp-A8h]
  unsigned int **v51; // [rsp+28h] [rbp-A8h]
  unsigned int *v52; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+30h] [rbp-A0h]
  _BYTE *i; // [rsp+38h] [rbp-98h] BYREF
  const char *v55; // [rsp+40h] [rbp-90h] BYREF
  char v56; // [rsp+60h] [rbp-70h]
  char v57; // [rsp+61h] [rbp-6Fh]
  _QWORD v58[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v59; // [rsp+90h] [rbp-40h]

  v5 = a5;
  v8 = a5;
  v9 = *(_BYTE *)(a5 + 140) == 12;
  i = a3;
  if ( v9 )
  {
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v10 = sub_91A3A0(*(_QWORD *)(*a1 + 32LL) + 8LL, *(_QWORD *)(v5 + 160), (__int64)a3, a4);
  v11 = a4;
  v53 = v10;
  v48 = *(_QWORD *)(a2 + 8);
  if ( (unsigned __int8)sub_91B6F0(a4)
    || (v31 = *(_DWORD *)(*((_QWORD *)i + 1) + 8LL) >> 8, v32 = sub_91B6E0(v11), v32 <= v31) )
  {
    v12 = (unsigned int **)a1[1];
    v13 = i;
    goto LABEL_5;
  }
  v33 = sub_BCCE00(a1[2], v32);
  v34 = i;
  v57 = 1;
  v50 = v33;
  v12 = (unsigned int **)a1[1];
  v35 = v33;
  v55 = "idx.ext";
  v56 = 3;
  if ( v33 == *((_QWORD *)i + 1) )
  {
    v13 = i;
    goto LABEL_22;
  }
  v36 = (__int64)v12[10];
  v37 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v36 + 120LL);
  if ( v37 == sub_920130 )
  {
    if ( *i > 0x15u )
    {
LABEL_24:
      v44 = v12;
      v46 = v34;
      v59 = 257;
      v39 = sub_BD2C40(72, unk_3F10A14);
      v40 = v44;
      v13 = (_BYTE *)v39;
      if ( v39 )
      {
        sub_B515B0(v39, v46, v50, v58, 0, 0);
        v40 = v44;
      }
      v51 = v40;
      (*(void (__fastcall **)(unsigned int *, _BYTE *, const char **, unsigned int *, unsigned int *))(*(_QWORD *)v40[11] + 16LL))(
        v40[11],
        v13,
        &v55,
        v40[7],
        v40[8]);
      v41 = *v51;
      v47 = (__int64)&(*v51)[4 * *((unsigned int *)v51 + 2)];
      if ( *v51 != (unsigned int *)v47 )
      {
        do
        {
          v52 = v41;
          sub_B99FD0(v13, *v41, *((_QWORD *)v41 + 1));
          v41 = v52 + 4;
        }
        while ( (unsigned int *)v47 != v52 + 4 );
      }
      v12 = (unsigned int **)a1[1];
      goto LABEL_22;
    }
    v43 = v12;
    if ( (unsigned __int8)sub_AC4810(39) )
      v38 = sub_ADAB70(39, i, v35, 0);
    else
      v38 = sub_AA93C0(39, i, v50);
    v34 = i;
    v12 = v43;
    v13 = (_BYTE *)v38;
  }
  else
  {
    v45 = v12;
    v42 = v37(v36, 39u, i, v50);
    v12 = v45;
    v34 = i;
    v13 = (_BYTE *)v42;
  }
  if ( !v13 )
    goto LABEL_24;
  v12 = (unsigned int **)a1[1];
LABEL_22:
  i = v13;
LABEL_5:
  v58[0] = "sub.ptr.neg";
  v59 = 259;
  v49 = v12;
  v14 = (_BYTE *)sub_AD6530(*((_QWORD *)v13 + 1));
  for ( i = (_BYTE *)sub_929DE0(v49, v14, v13, (__int64)v58, 0, 0); *(_BYTE *)(v8 + 140) == 12; v8 = *(_QWORD *)(v8 + 160) )
    ;
  v15 = *(_QWORD *)(v8 + 160);
  for ( j = *(_BYTE *)(v15 + 140); j == 12; j = *(_BYTE *)(v15 + 140) )
    v15 = *(_QWORD *)(v15 + 160);
  if ( j == 1 || *(_BYTE *)(v53 + 8) == 13 )
  {
    v17 = *(_DWORD *)(v48 + 8);
    v18 = sub_BCB2B0(a1[2]);
    v19 = sub_BCE760(v18, v17 >> 8);
    v20 = (unsigned int **)a1[1];
    v59 = 257;
    v21 = sub_929600(v20, 0x31u, a2, v19, (__int64)v58, 0, (unsigned int)v55, 0);
    v22 = a1[2];
    v23 = (unsigned int **)a1[1];
    v24 = v21;
    v58[0] = "sub.ptr";
    v59 = 259;
    v25 = sub_BCB2B0(v22);
    v26 = sub_921130(v23, v25, v24, &i, 1, (__int64)v58, 0);
    v27 = (unsigned int **)a1[1];
    v28 = *(_QWORD *)(a2 + 8);
    v59 = 257;
    return sub_929600(v27, 0x31u, v26, v28, (__int64)v58, 0, (unsigned int)v55, 0);
  }
  else
  {
    v30 = (unsigned int **)a1[1];
    v58[0] = "sub.ptr";
    v59 = 259;
    return sub_921130(v30, v53, a2, &i, 1, (__int64)v58, 3u);
  }
}
