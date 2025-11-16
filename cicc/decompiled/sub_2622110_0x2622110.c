// Function: sub_2622110
// Address: 0x2622110
//
__int64 __fastcall sub_2622110(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // r9
  unsigned int v9; // edx
  _QWORD *v10; // rcx
  int v12; // ecx
  int v13; // ecx
  _QWORD *v14; // r8
  unsigned __int64 v15; // rbx
  __int64 v16; // r9
  _QWORD *v17; // r15
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rsi
  _BYTE *v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // r15
  unsigned int v24; // esi
  __int64 v25; // r8
  _QWORD *v26; // r10
  int v27; // r11d
  unsigned int v28; // eax
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  __int64 *v31; // r15
  __int64 v32; // rsi
  __int64 v33; // r8
  unsigned __int64 v34; // rsi
  _QWORD *v35; // r14
  _QWORD *v36; // r13
  unsigned __int64 v37; // rdi
  _QWORD *v38; // rax
  _QWORD *v39; // rsi
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rcx
  _BYTE *v42; // r8
  bool v43; // r10
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  bool v46; // r10
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // rdi
  _BYTE *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // r10
  _BYTE *v53; // rdi
  _BYTE *v54; // rax
  _QWORD *v55; // r10
  int v56; // eax
  int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // [rsp+8h] [rbp-A8h]
  char v61; // [rsp+14h] [rbp-9Ch]
  __int64 v62; // [rsp+18h] [rbp-98h]
  char v63; // [rsp+18h] [rbp-98h]
  __int64 v64; // [rsp+20h] [rbp-90h]
  _QWORD *v65; // [rsp+20h] [rbp-90h]
  __int64 v66; // [rsp+20h] [rbp-90h]
  __int64 *v67; // [rsp+28h] [rbp-88h]
  _QWORD *v68; // [rsp+28h] [rbp-88h]
  _QWORD *v69; // [rsp+28h] [rbp-88h]
  __int64 v70; // [rsp+38h] [rbp-78h] BYREF
  __int64 v71; // [rsp+48h] [rbp-68h] BYREF
  _QWORD *v72; // [rsp+50h] [rbp-60h] BYREF
  __int64 v73; // [rsp+58h] [rbp-58h]
  unsigned __int64 v74; // [rsp+60h] [rbp-50h]
  __int64 v75; // [rsp+68h] [rbp-48h]
  char v76; // [rsp+70h] [rbp-40h]

  v2 = a2;
  v4 = *a1;
  v70 = (__int64)a2;
  v72 = a2;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v5 = *(_DWORD *)(v4 + 152);
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 128);
    v71 = 0;
LABEL_76:
    sub_261CF40(v4 + 128, 2 * v5);
    goto LABEL_77;
  }
  v6 = *(_QWORD *)(v4 + 136);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v60 = v6 + 40LL * v9;
  v10 = *(_QWORD **)v60;
  if ( v2 == *(_QWORD **)v60 )
    return v60 + 8;
  while ( v10 != (_QWORD *)-4096LL )
  {
    if ( !v8 && v10 == (_QWORD *)-8192LL )
      v8 = v60;
    v9 = (v5 - 1) & (v7 + v9);
    v60 = v6 + 40LL * v9;
    v10 = *(_QWORD **)v60;
    if ( v2 == *(_QWORD **)v60 )
      return v60 + 8;
    ++v7;
  }
  v12 = *(_DWORD *)(v4 + 144);
  if ( !v8 )
    v8 = v60;
  ++*(_QWORD *)(v4 + 128);
  v13 = v12 + 1;
  v60 = v8;
  v71 = v8;
  if ( 4 * v13 >= 3 * v5 )
    goto LABEL_76;
  if ( v5 - *(_DWORD *)(v4 + 148) - v13 > v5 >> 3 )
    goto LABEL_14;
  sub_261CF40(v4 + 128, v5);
LABEL_77:
  sub_2618D80(v4 + 128, (__int64 *)&v72, &v71);
  v2 = v72;
  v60 = v71;
  v13 = *(_DWORD *)(v4 + 144) + 1;
LABEL_14:
  *(_DWORD *)(v4 + 144) = v13;
  if ( *(_QWORD *)v60 != -4096 )
    --*(_DWORD *)(v4 + 148);
  *(_QWORD *)v60 = v2;
  *(_QWORD *)(v60 + 8) = v73;
  *(_QWORD *)(v60 + 16) = v74;
  *(_QWORD *)(v60 + 24) = v75;
  *(_BYTE *)(v60 + 32) = v76;
  v14 = (_QWORD *)a1[1];
  v73 = 1;
  v15 = v70 & 0xFFFFFFFFFFFFFFFCLL | 1;
  v72 = &v72;
  v16 = (__int64)(v14 + 1);
  v74 = v15;
  v17 = (_QWORD *)v14[2];
  if ( !v17 )
  {
    v17 = v14 + 1;
    if ( v14[3] == v16 )
    {
      v46 = 1;
LABEL_65:
      v65 = v14;
      v68 = (_QWORD *)v16;
      v63 = v46;
      v20 = (_QWORD *)sub_22077B0(0x38u);
      v20[4] = v20 + 4;
      v47 = v74;
      v20[5] = 1;
      v20[6] = v47;
      sub_220F040(v63, (__int64)v20, v17, v68);
      ++v65[5];
      v16 = a1[1] + 8;
      goto LABEL_25;
    }
LABEL_68:
    v66 = (__int64)(v14 + 1);
    v69 = v14;
    v48 = sub_220EF80((__int64)v17);
    v14 = v69;
    v16 = v66;
    if ( *(_QWORD *)(v48 + 48) >= v15 )
    {
      v17 = (_QWORD *)v48;
      goto LABEL_24;
    }
LABEL_63:
    v46 = 1;
    if ( (_QWORD *)v16 != v17 )
      v46 = v15 < v17[6];
    goto LABEL_65;
  }
  while ( 1 )
  {
    v18 = v17[6];
    v19 = (_QWORD *)v17[3];
    if ( v18 > v15 )
      v19 = (_QWORD *)v17[2];
    if ( !v19 )
      break;
    v17 = v19;
  }
  if ( v15 < v18 )
  {
    if ( v17 == (_QWORD *)v14[3] )
      goto LABEL_63;
    goto LABEL_68;
  }
  if ( v15 > v18 )
    goto LABEL_63;
LABEL_24:
  v20 = v17;
LABEL_25:
  v21 = 0;
  if ( v20 != (_QWORD *)v16 )
  {
    v21 = v20 + 4;
    if ( (v20[5] & 1) == 0 )
    {
      v21 = (_BYTE *)v20[4];
      if ( (v21[8] & 1) == 0 )
      {
        v22 = *(_QWORD *)v21;
        if ( (*(_BYTE *)(*(_QWORD *)v21 + 8LL) & 1) != 0 )
        {
          v21 = *(_BYTE **)v21;
        }
        else
        {
          v49 = *(_BYTE **)v22;
          if ( (*(_BYTE *)(*(_QWORD *)v22 + 8LL) & 1) == 0 )
          {
            v50 = sub_261C730(v49);
            *v51 = v50;
            v49 = v50;
          }
          *(_QWORD *)v21 = v49;
          v21 = v49;
        }
        v20[4] = v21;
      }
    }
  }
  v23 = a1[2];
  v24 = *(_DWORD *)(v23 + 24);
  if ( !v24 )
  {
    v72 = 0;
    ++*(_QWORD *)v23;
LABEL_98:
    v24 *= 2;
    goto LABEL_99;
  }
  v25 = *(_QWORD *)(v23 + 8);
  v26 = 0;
  v27 = 1;
  v28 = (v24 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
  v29 = (_QWORD *)(v25 + 40LL * v28);
  v30 = *v29;
  if ( *v29 == v70 )
  {
LABEL_33:
    v67 = (__int64 *)v29[3];
    if ( v67 == (__int64 *)v29[2] )
      return v60 + 8;
    v31 = (__int64 *)v29[2];
    while ( 1 )
    {
      v32 = *v31;
      v73 = 1;
      v33 = a1[1];
      v34 = v32 & 0xFFFFFFFFFFFFFFFCLL;
      v72 = &v72;
      v74 = v34;
      v35 = *(_QWORD **)(v33 + 16);
      v36 = (_QWORD *)(v33 + 8);
      if ( v35 )
        break;
      v35 = (_QWORD *)(v33 + 8);
      if ( v36 != *(_QWORD **)(v33 + 24) )
        goto LABEL_59;
      v43 = 1;
LABEL_57:
      v62 = v33;
      v61 = v43;
      v39 = (_QWORD *)sub_22077B0(0x38u);
      v39[4] = v39 + 4;
      v44 = v74;
      v39[5] = 1;
      v39[6] = v44;
      sub_220F040(v61, (__int64)v39, v35, v36);
      ++*(_QWORD *)(v62 + 40);
LABEL_44:
      v40 = 0;
      if ( v39 != v36 )
      {
        v40 = (unsigned __int64)(v39 + 4);
        if ( (v39[5] & 1) == 0 )
        {
          v40 = v39[4];
          if ( (*(_BYTE *)(v40 + 8) & 1) == 0 )
          {
            v41 = *(_QWORD *)v40;
            if ( (*(_BYTE *)(*(_QWORD *)v40 + 8LL) & 1) != 0 )
            {
              v40 = *(_QWORD *)v40;
            }
            else
            {
              v42 = *(_BYTE **)v41;
              if ( (*(_BYTE *)(*(_QWORD *)v41 + 8LL) & 1) == 0 )
              {
                v52 = *(_QWORD *)v42;
                if ( (*(_BYTE *)(*(_QWORD *)v42 + 8LL) & 1) != 0 )
                {
                  v42 = *(_BYTE **)v42;
                }
                else
                {
                  v53 = *(_BYTE **)v52;
                  if ( (*(_BYTE *)(*(_QWORD *)v52 + 8LL) & 1) == 0 )
                  {
                    v54 = sub_261C730(v53);
                    *v55 = v54;
                    v53 = v54;
                  }
                  *(_QWORD *)v42 = v53;
                  v42 = v53;
                }
                *(_QWORD *)v41 = v42;
              }
              *(_QWORD *)v40 = v42;
              v40 = (unsigned __int64)v42;
            }
            v39[4] = v40;
          }
        }
      }
      if ( v21 != (_BYTE *)v40 )
      {
        *(_QWORD *)(*(_QWORD *)v21 + 8LL) = v40 | *(_QWORD *)(*(_QWORD *)v21 + 8LL) & 1LL;
        *(_QWORD *)v21 = *(_QWORD *)v40;
        *(_QWORD *)(v40 + 8) &= ~1uLL;
        *(_QWORD *)v40 = v21;
      }
      if ( v67 == ++v31 )
        return v60 + 8;
    }
    while ( 1 )
    {
      v37 = v35[6];
      v38 = (_QWORD *)v35[3];
      if ( v37 > v34 )
        v38 = (_QWORD *)v35[2];
      if ( !v38 )
        break;
      v35 = v38;
    }
    if ( v34 < v37 )
    {
      if ( *(_QWORD **)(v33 + 24) != v35 )
      {
LABEL_59:
        v64 = v33;
        v45 = sub_220EF80((__int64)v35);
        v33 = v64;
        if ( v34 <= *(_QWORD *)(v45 + 48) )
        {
          v35 = (_QWORD *)v45;
LABEL_43:
          v39 = v35;
          goto LABEL_44;
        }
      }
    }
    else if ( v34 <= v37 )
    {
      goto LABEL_43;
    }
    v43 = 1;
    if ( v35 != v36 )
      v43 = v34 < v35[6];
    goto LABEL_57;
  }
  while ( v30 != -4096 )
  {
    if ( !v26 && v30 == -8192 )
      v26 = v29;
    v28 = (v24 - 1) & (v27 + v28);
    v29 = (_QWORD *)(v25 + 40LL * v28);
    v30 = *v29;
    if ( v70 == *v29 )
      goto LABEL_33;
    ++v27;
  }
  if ( !v26 )
    v26 = v29;
  v72 = v26;
  v56 = *(_DWORD *)(v23 + 16);
  ++*(_QWORD *)v23;
  v57 = v56 + 1;
  if ( 4 * (v56 + 1) >= 3 * v24 )
    goto LABEL_98;
  if ( v24 - *(_DWORD *)(v23 + 20) - v57 <= v24 >> 3 )
  {
LABEL_99:
    sub_261D190(v23, v24);
    sub_2618CC0(v23, &v70, &v72);
    v57 = *(_DWORD *)(v23 + 16) + 1;
  }
  *(_DWORD *)(v23 + 16) = v57;
  v58 = v72;
  if ( *v72 != -4096 )
    --*(_DWORD *)(v23 + 20);
  v59 = v70;
  *(_OWORD *)(v58 + 1) = 0;
  *v58 = v59;
  *(_OWORD *)(v58 + 3) = 0;
  return v60 + 8;
}
