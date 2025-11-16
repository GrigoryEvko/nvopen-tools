// Function: sub_2473100
// Address: 0x2473100
//
void __fastcall sub_2473100(__int64 *a1, unsigned __int8 *a2, char a3)
{
  __int64 v3; // r14
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r12
  int v15; // ebx
  __int64 v16; // r15
  unsigned __int8 *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // r10
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  _BYTE *v30; // rbx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  __int64 v36; // r15
  unsigned __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned int **v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r12
  __int64 v45; // r15
  _BYTE *v46; // r11
  bool v47; // al
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // [rsp+0h] [rbp-190h]
  __int64 v54; // [rsp+8h] [rbp-188h]
  _BYTE *v55; // [rsp+10h] [rbp-180h]
  __int64 v56; // [rsp+10h] [rbp-180h]
  __int64 v57; // [rsp+30h] [rbp-160h]
  __int64 v58; // [rsp+30h] [rbp-160h]
  unsigned int v60; // [rsp+38h] [rbp-158h]
  __int64 v61; // [rsp+38h] [rbp-158h]
  _QWORD *v62; // [rsp+38h] [rbp-158h]
  __int64 v63; // [rsp+40h] [rbp-150h]
  _QWORD v64[4]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v65; // [rsp+70h] [rbp-120h]
  _BYTE *v66; // [rsp+80h] [rbp-110h] BYREF
  __int64 v67; // [rsp+88h] [rbp-108h]
  _BYTE v68[64]; // [rsp+90h] [rbp-100h] BYREF
  unsigned int *v69[24]; // [rsp+D0h] [rbp-C0h] BYREF

  v3 = (__int64)a2;
  sub_23D0AB0((__int64)v69, (__int64)a2, 0, 0, 0);
  v4 = *a2;
  if ( v4 == 40 )
  {
    v5 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v5 = 0;
    if ( v4 != 85 )
    {
      v5 = 64;
      if ( v4 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v6 = sub_BD2BC0((__int64)a2);
  v8 = v6 + v7;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v8 >> 4) )
LABEL_53:
      BUG();
LABEL_10:
    v12 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v8 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_53;
  v9 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v10 = sub_BD2BC0((__int64)a2);
  v12 = 32LL * (unsigned int)(*(_DWORD *)(v10 + v11 - 4) - v9);
LABEL_11:
  v13 = (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v5 - v12) >> 5;
  v14 = (unsigned int)(v13 - 1);
  v57 = *(_QWORD *)&a2[32 * (v14 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  if ( (_BYTE)qword_4FE84C8 )
    sub_2472230((__int64)a1, *(_QWORD *)&a2[32 * (v14 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))], (__int64)a2);
  v15 = v13 - 2;
  if ( a3 )
    LODWORD(v14) = v15;
  v16 = 0;
  v66 = v68;
  v67 = 0x800000000LL;
  if ( (int)v14 > 0 )
  {
    do
    {
      if ( (a2[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v17 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v20 = sub_246F3F0((__int64)a1, *(_QWORD *)&v17[v16]);
      v21 = (unsigned int)v67;
      v22 = (unsigned int)v67 + 1LL;
      if ( v22 > HIDWORD(v67) )
      {
        sub_C8D5F0((__int64)&v66, v68, v22, 8u, v18, v19);
        v21 = (unsigned int)v67;
      }
      v16 += 32;
      *(_QWORD *)&v66[8 * v21] = v20;
      LODWORD(v67) = v67 + 1;
    }
    while ( 32LL * (int)v14 != v16 );
    v3 = (__int64)a2;
  }
  v23 = *(_QWORD *)(*(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)) + 8LL);
  v54 = sub_BCDA70(*(__int64 **)(v23 + 24), (int)v14 * *(_DWORD *)(v23 + 32));
  v26 = sub_2463540(a1, v54);
  if ( a3 )
  {
    v27 = *(_QWORD *)(v3 + 32 * ((unsigned int)v14 - (unsigned __int64)(*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
    v28 = (unsigned int)v67;
    v29 = (unsigned int)v67 + 1LL;
    if ( v29 > HIDWORD(v67) )
    {
      v62 = v26;
      sub_C8D5F0((__int64)&v66, v68, v29, 8u, v24, v25);
      v28 = (unsigned int)v67;
      v26 = v62;
    }
    *(_QWORD *)&v66[8 * v28] = v27;
    LODWORD(v67) = v67 + 1;
  }
  v30 = sub_2466120((__int64)a1, v57, v69, (__int64)v26, 0x100u, 1);
  v33 = (unsigned int)v67;
  v53 = v34;
  v35 = (unsigned int)v67 + 1LL;
  if ( v35 > HIDWORD(v67) )
  {
    sub_C8D5F0((__int64)&v66, v68, v35, 8u, v31, v32);
    v33 = (unsigned int)v67;
  }
  HIDWORD(v63) = 0;
  *(_QWORD *)&v66[8 * v33] = v30;
  v65 = 257;
  v36 = (__int64)v66;
  v37 = (unsigned int)(v67 + 1);
  v38 = *(_QWORD *)(v3 - 32);
  LODWORD(v67) = v67 + 1;
  if ( !v38 || *(_BYTE *)v38 || *(_QWORD *)(v38 + 24) != *(_QWORD *)(v3 + 80) )
    BUG();
  v60 = *(_DWORD *)(v38 + 36);
  v39 = sub_BCB120((_QWORD *)v69[9]);
  v40 = sub_B35180((__int64)v69, v39, v60, v36, v37, v63, (__int64)v64);
  v41 = (unsigned int **)v3;
  sub_246EF60((__int64)a1, v3, v40);
  if ( *(_DWORD *)(a1[1] + 4) )
  {
    if ( (int)v14 > 0 )
    {
      v42 = (int)v14;
      v43 = 0;
      v44 = 0;
      v58 = v42;
      while ( 1 )
      {
        while ( 1 )
        {
          v45 = *(_QWORD *)(v3 + 32 * (v43 - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
          v61 = sub_246F3F0((__int64)a1, v45);
          if ( !*(_DWORD *)(a1[1] + 4) )
            goto LABEL_36;
          v46 = (_BYTE *)sub_246EE10((__int64)a1, v45);
          if ( !*(_DWORD *)(a1[1] + 4) )
            goto LABEL_36;
          if ( v44 )
            break;
          v44 = (__int64)v46;
LABEL_36:
          if ( ++v43 == v58 )
            goto LABEL_43;
        }
        if ( *v46 <= 0x15u )
        {
          v55 = v46;
          v47 = sub_AC30F0((__int64)v46);
          v46 = v55;
          if ( v47 )
            goto LABEL_36;
        }
        ++v43;
        v65 = 257;
        v56 = (__int64)v46;
        v48 = sub_2465600((__int64)a1, v61, (__int64)v69, (__int64)v64);
        v65 = 257;
        v44 = sub_B36550(v69, v48, v56, v44, (__int64)v64, 0);
        if ( v43 == v58 )
          goto LABEL_43;
      }
    }
    v44 = 0;
LABEL_43:
    v49 = sub_B2BEC0(*a1);
    v41 = (unsigned int **)v54;
    v50 = sub_9C6480(v49, v54);
    v52 = a1[1];
    v64[0] = v50;
    v64[1] = v51;
    if ( *(_DWORD *)(v52 + 4) )
    {
      v41 = v69;
      sub_24677C0(a1, (__int64)v69, v44, v53, v50, v51, byte_4FE8EA9);
    }
  }
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  sub_F94A20(v69, (__int64)v41);
}
