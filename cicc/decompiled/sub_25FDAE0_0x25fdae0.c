// Function: sub_25FDAE0
// Address: 0x25fdae0
//
unsigned __int8 *__fastcall sub_25FDAE0(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rax
  bool v4; // zf
  unsigned __int8 *v5; // r12
  __int64 v6; // r10
  unsigned __int64 v7; // rax
  int v8; // r15d
  unsigned int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r12
  unsigned int v13; // r8d
  int *v14; // rdx
  int v15; // edi
  _BYTE *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // r8d
  unsigned int v23; // edi
  unsigned int *v24; // r9
  int v25; // r9d
  unsigned int v26; // r11d
  int v27; // r10d
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 *v30; // r14
  __int64 v31; // r15
  __int64 v32; // r13
  unsigned __int8 *v33; // rax
  __int64 v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // r13
  __int64 v37; // rax
  __int64 *v38; // r14
  __int64 v39; // rsi
  int v41; // edx
  int v42; // r10d
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  int v45; // edx
  __int64 v46; // r15
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r13
  __int64 v51; // rax
  int v52; // r13d
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  int v56; // ebx
  __int64 *v57; // rax
  unsigned int v58; // eax
  int v59; // r9d
  int v60; // r10d
  __int64 v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+18h] [rbp-98h]
  __int64 v64; // [rsp+18h] [rbp-98h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+18h] [rbp-98h]
  __int64 v67; // [rsp+20h] [rbp-90h]
  __int64 v68; // [rsp+20h] [rbp-90h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  unsigned __int64 v70; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v71; // [rsp+38h] [rbp-78h]
  _BYTE *v72; // [rsp+40h] [rbp-70h]
  __int64 v73[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v74; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 296);
  v4 = *(_BYTE *)(a2 + 128) == 0;
  v70 = 0;
  v5 = *(unsigned __int8 **)(a2 + 240);
  v71 = 0;
  v6 = *(_QWORD *)(v3 + 56);
  v72 = 0;
  v69 = v3;
  if ( !v4 )
    goto LABEL_2;
  v45 = *v5;
  v46 = *(_QWORD *)(v6 + 104);
  if ( v45 == 40 )
  {
    v68 = v6;
    v58 = sub_B491D0((__int64)v5);
    v6 = v68;
    v47 = 32LL * v58;
  }
  else
  {
    v47 = 0;
    if ( v45 != 85 )
    {
      v47 = 64;
      if ( v45 != 34 )
        BUG();
    }
  }
  if ( (v5[7] & 0x80u) == 0 )
    goto LABEL_65;
  v67 = v6;
  v48 = sub_BD2BC0((__int64)v5);
  v6 = v67;
  v50 = v48 + v49;
  if ( (v5[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v50 >> 4) )
LABEL_79:
      BUG();
LABEL_65:
    v55 = 0;
    goto LABEL_66;
  }
  v51 = sub_BD2BC0((__int64)v5);
  v6 = v67;
  if ( !(unsigned int)((v50 - v51) >> 4) )
    goto LABEL_65;
  if ( (v5[7] & 0x80u) == 0 )
    goto LABEL_79;
  v52 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
  if ( (v5[7] & 0x80u) == 0 )
    BUG();
  v53 = sub_BD2BC0((__int64)v5);
  v6 = v67;
  v55 = 32LL * (unsigned int)(*(_DWORD *)(v53 + v54 - 4) - v52);
LABEL_66:
  if ( (unsigned int)((32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF) - 32 - v47 - v55) >> 5) == v46 )
  {
    *((_QWORD *)v5 + 10) = *(_QWORD *)(v6 + 24);
    sub_AC2B30((__int64)(v5 - 32), v6);
    goto LABEL_39;
  }
LABEL_2:
  v7 = *(_QWORD *)(v6 + 104);
  if ( v7 )
  {
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = (__int64)v5;
    v12 = v6;
    while ( v7 - 1 != v10 || *(_DWORD *)(v69 + 152) <= 1u )
    {
      v18 = *(unsigned int *)(a2 + 88);
      v19 = *(_QWORD *)(a2 + 72);
      if ( !(_DWORD)v18 )
        goto LABEL_14;
      v13 = (v18 - 1) & v8;
      v14 = (int *)(v19 + 8LL * v13);
      v15 = *v14;
      if ( *v14 != v9 )
      {
        v41 = 1;
        while ( v15 != -1 )
        {
          v42 = v41 + 1;
          v13 = (v18 - 1) & (v41 + v13);
          v14 = (int *)(v19 + 8LL * v13);
          v15 = *v14;
          if ( *v14 == v9 )
            goto LABEL_5;
          v41 = v42;
        }
LABEL_14:
        v20 = *(unsigned int *)(a2 + 192);
        v21 = *(_QWORD *)(a2 + 176);
        if ( (_DWORD)v20 )
        {
          v22 = v20 - 1;
          v23 = (v20 - 1) & v8;
          v24 = (unsigned int *)(v21 + 16LL * v23);
          v19 = *v24;
          if ( (_DWORD)v19 == v9 )
          {
LABEL_16:
            v17 = *((_QWORD *)v24 + 1);
            v16 = v71;
            v73[0] = v17;
            if ( v71 != v72 )
              goto LABEL_7;
LABEL_17:
            v63 = v11;
            sub_928380((__int64)&v70, v16, v73);
            v11 = v63;
            goto LABEL_10;
          }
          v25 = *v24;
          v26 = (v20 - 1) & v8;
          v27 = 1;
          while ( v25 != -1 )
          {
            v26 = v22 & (v27 + v26);
            v25 = *(_DWORD *)(v21 + 16LL * v26);
            if ( v25 == v9 )
            {
              v59 = 1;
              while ( (_DWORD)v19 != -1 )
              {
                v60 = v59 + 1;
                v23 = v22 & (v59 + v23);
                v24 = (unsigned int *)(v21 + 16LL * v23);
                LODWORD(v19) = *v24;
                if ( *v24 == v9 )
                  goto LABEL_16;
                v59 = v60;
              }
              v24 = (unsigned int *)(v21 + 16 * v20);
              goto LABEL_16;
            }
            ++v27;
          }
        }
        if ( (*(_BYTE *)(v12 + 2) & 1) != 0 )
        {
          v65 = v11;
          sub_B2C6D0(v12, v19, v21, v11);
          v11 = v65;
        }
        v64 = v11;
        v17 = sub_AC9EC0(*(__int64 ***)(*(_QWORD *)(v12 + 96) + 40 * v10 + 8));
LABEL_23:
        v16 = v71;
        v73[0] = v17;
        v11 = v64;
        if ( v71 != v72 )
        {
LABEL_7:
          if ( v16 )
          {
            *(_QWORD *)v16 = v17;
            v16 = v71;
          }
          v71 = v16 + 8;
          goto LABEL_10;
        }
        goto LABEL_17;
      }
LABEL_5:
      if ( v14 == (int *)(v19 + 8 * v18) )
        goto LABEL_14;
      v16 = v71;
      v17 = *(_QWORD *)(v11 + 32 * ((unsigned int)v14[1] - (unsigned __int64)(*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
      v73[0] = v17;
      if ( v71 != v72 )
        goto LABEL_7;
      v66 = v11;
      sub_9281F0((__int64)&v70, v71, v73);
      v11 = v66;
LABEL_10:
      v7 = *(_QWORD *)(v12 + 104);
      v8 += 37;
      v10 = ++v9;
      if ( v9 >= v7 )
      {
        v6 = v12;
        v5 = (unsigned __int8 *)v11;
        goto LABEL_27;
      }
    }
    v28 = *(unsigned int *)(a2 + 28);
    v64 = v11;
    v29 = sub_BCB2D0(*a1);
    v17 = sub_ACD640(v29, v28, 0);
    goto LABEL_23;
  }
LABEL_27:
  v30 = (__int64 *)v70;
  v74 = 257;
  v31 = *(_QWORD *)(v6 + 24);
  v32 = (__int64)&v71[-v70] >> 3;
  v62 = v6;
  v33 = (unsigned __int8 *)sub_BD2C40(88, (int)v32 + 1);
  v34 = (__int64)(v5 + 24);
  v5 = v33;
  if ( v33 )
  {
    sub_B44260((__int64)v33, **(_QWORD **)(v31 + 16), 56, (v32 + 1) & 0x7FFFFFF, v34, 0);
    *((_QWORD *)v5 + 9) = 0;
    sub_B4A290((__int64)v5, v31, v62, v30, v32, (__int64)v73, 0, 0);
  }
  v35 = *(_QWORD *)(a2 + 8);
  v36 = *(_QWORD **)(a2 + 240);
  if ( *(_QWORD **)(v35 + 16) == v36 )
    *(_QWORD *)(v35 + 16) = v5;
  v37 = *(_QWORD *)(a2 + 16);
  if ( *(_QWORD **)(v37 + 16) == v36 )
    *(_QWORD *)(v37 + 16) = v5;
  v38 = (__int64 *)(v5 + 48);
  v39 = *(_QWORD *)(*(_QWORD *)(a2 + 240) + 48LL);
  v73[0] = v39;
  if ( v39 )
  {
    sub_B96E90((__int64)v73, v39, 1);
    if ( v38 == v73 )
    {
      if ( v73[0] )
        sub_B91220((__int64)v73, v73[0]);
      goto LABEL_37;
    }
    v43 = *((_QWORD *)v5 + 6);
    if ( !v43 )
    {
LABEL_49:
      v44 = (unsigned __int8 *)v73[0];
      *((_QWORD *)v5 + 6) = v73[0];
      if ( v44 )
        sub_B976B0((__int64)v73, v44, (__int64)(v5 + 48));
      goto LABEL_37;
    }
LABEL_48:
    sub_B91220((__int64)(v5 + 48), v43);
    goto LABEL_49;
  }
  if ( v38 != v73 )
  {
    v43 = *((_QWORD *)v5 + 6);
    if ( v43 )
      goto LABEL_48;
  }
LABEL_37:
  sub_BD84D0((__int64)v36, (__int64)v5);
  sub_B43D60(v36);
  *(_QWORD *)(a2 + 240) = v5;
  if ( *(_BYTE *)(v69 + 316) )
  {
    v56 = *(_DWORD *)(v69 + 312);
    v57 = (__int64 *)sub_BD5C60((__int64)v5);
    *((_QWORD *)v5 + 9) = sub_A7A090((__int64 *)v5 + 9, v57, v56 + 1, 74);
  }
LABEL_39:
  if ( v70 )
    j_j___libc_free_0(v70);
  return v5;
}
