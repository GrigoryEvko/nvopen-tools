// Function: sub_92AA70
// Address: 0x92aa70
//
__int64 __fastcall sub_92AA70(_QWORD *a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v9; // rbx
  bool v10; // zf
  __int64 v11; // r13
  __int64 v12; // rax
  char i; // dl
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int **v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned int **v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int **v24; // rdi
  __int64 v25; // rcx
  unsigned int **v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rax
  _BYTE *v30; // r10
  __int64 v31; // r11
  unsigned int **v32; // rax
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // rsi
  unsigned int *v38; // rcx
  __int64 v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-B8h]
  __int64 v41; // [rsp+18h] [rbp-B8h]
  __int64 v42; // [rsp+18h] [rbp-B8h]
  _BYTE *v43; // [rsp+20h] [rbp-B0h]
  __int64 v44; // [rsp+20h] [rbp-B0h]
  _BYTE *v45; // [rsp+20h] [rbp-B0h]
  unsigned int *v46; // [rsp+20h] [rbp-B0h]
  unsigned int v47; // [rsp+28h] [rbp-A8h]
  unsigned int **v48; // [rsp+28h] [rbp-A8h]
  _BYTE *v49; // [rsp+28h] [rbp-A8h]
  __int64 v50; // [rsp+30h] [rbp-A0h]
  _BYTE *v51; // [rsp+38h] [rbp-98h] BYREF
  const char *v52; // [rsp+40h] [rbp-90h] BYREF
  char v53; // [rsp+60h] [rbp-70h]
  char v54; // [rsp+61h] [rbp-6Fh]
  _QWORD v55[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v56; // [rsp+90h] [rbp-40h]

  v5 = a5;
  v9 = a5;
  v10 = *(_BYTE *)(a5 + 140) == 12;
  v51 = a3;
  if ( v10 )
  {
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v11 = sub_91A3A0(*(_QWORD *)(*a1 + 32LL) + 8LL, *(_QWORD *)(v5 + 160), (__int64)a3, a4);
  v50 = *(_QWORD *)(a2 + 8);
  if ( !(unsigned __int8)sub_91B6F0(a4) )
  {
    v47 = *(_DWORD *)(*((_QWORD *)v51 + 1) + 8LL) >> 8;
    v28 = sub_91B6E0(a4);
    if ( v28 > v47 )
    {
      v29 = sub_BCCE00(a1[2], v28);
      v30 = v51;
      v54 = 1;
      v31 = v29;
      v32 = (unsigned int **)a1[1];
      v52 = "idx.ext";
      v53 = 3;
      v48 = v32;
      if ( v31 == *((_QWORD *)v51 + 1) )
      {
        v35 = v51;
        goto LABEL_21;
      }
      v33 = (__int64)v32[10];
      v34 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v33 + 120LL);
      if ( v34 == sub_920130 )
      {
        if ( *v51 > 0x15u )
          goto LABEL_23;
        v39 = v31;
        if ( (unsigned __int8)sub_AC4810(39) )
          v35 = (_BYTE *)sub_ADAB70(39, v51, v39, 0);
        else
          v35 = (_BYTE *)sub_AA93C0(39, v51, v39);
        v30 = v51;
        v31 = v39;
      }
      else
      {
        v42 = v31;
        v35 = (_BYTE *)v34(v33, 39u, v51, v31);
        v31 = v42;
        v30 = v51;
      }
      if ( v35 )
      {
LABEL_21:
        v51 = v35;
        goto LABEL_6;
      }
LABEL_23:
      v40 = v31;
      v43 = v30;
      v56 = 257;
      v36 = sub_BD2C40(72, unk_3F10A14);
      if ( v36 )
      {
        v37 = v43;
        v44 = v36;
        sub_B515B0(v36, v37, v40, v55, 0, 0);
        v36 = v44;
      }
      v45 = (_BYTE *)v36;
      (*(void (__fastcall **)(unsigned int *, __int64, const char **, unsigned int *, unsigned int *))(*(_QWORD *)v48[11] + 16LL))(
        v48[11],
        v36,
        &v52,
        v48[7],
        v48[8]);
      v38 = *v48;
      v41 = (__int64)&(*v48)[4 * *((unsigned int *)v48 + 2)];
      v35 = v45;
      if ( *v48 != (unsigned int *)v41 )
      {
        do
        {
          v46 = v38;
          v49 = v35;
          sub_B99FD0(v35, *v38, *((_QWORD *)v38 + 1));
          v35 = v49;
          v38 = v46 + 4;
        }
        while ( (unsigned int *)v41 != v46 + 4 );
      }
      goto LABEL_21;
    }
  }
LABEL_6:
  while ( *(_BYTE *)(v9 + 140) == 12 )
    v9 = *(_QWORD *)(v9 + 160);
  v12 = *(_QWORD *)(v9 + 160);
  for ( i = *(_BYTE *)(v12 + 140); i == 12; i = *(_BYTE *)(v12 + 140) )
    v12 = *(_QWORD *)(v12 + 160);
  if ( i == 1 || *(_BYTE *)(v11 + 8) == 13 )
  {
    v14 = *(_DWORD *)(v50 + 8);
    v15 = sub_BCB2B0(a1[2]);
    v16 = sub_BCE760(v15, v14 >> 8);
    v17 = (unsigned int **)a1[1];
    v56 = 257;
    v18 = sub_929600(v17, 0x31u, a2, v16, (__int64)v55, 0, (unsigned int)v52, 0);
    v19 = a1[2];
    v20 = (unsigned int **)a1[1];
    v21 = v18;
    v55[0] = "add.ptr";
    v56 = 259;
    v22 = sub_BCB2B0(v19);
    v23 = sub_921130(v20, v22, v21, &v51, 1, (__int64)v55, 0);
    v24 = (unsigned int **)a1[1];
    v25 = *(_QWORD *)(a2 + 8);
    v56 = 257;
    return sub_929600(v24, 0x31u, v23, v25, (__int64)v55, 0, (unsigned int)v52, 0);
  }
  else
  {
    v27 = (unsigned int **)a1[1];
    v55[0] = "add.ptr";
    v56 = 259;
    return sub_921130(v27, v11, a2, &v51, 1, (__int64)v55, 3u);
  }
}
