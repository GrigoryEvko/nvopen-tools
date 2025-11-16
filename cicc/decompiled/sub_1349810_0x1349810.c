// Function: sub_1349810
// Address: 0x1349810
//
unsigned __int64 __fastcall sub_1349810(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r9
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r14
  _QWORD *v7; // r13
  unsigned __int64 v8; // r11
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  bool v11; // zf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r15
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // rsi
  unsigned __int64 *v21; // rdi
  __int64 v22; // rbx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r12
  char v25; // dl
  __int64 v26; // rax
  int v27; // eax
  _QWORD *v28; // r9
  __int64 v29; // r12
  _QWORD *v30; // r13
  __int64 v31; // rbx
  __int64 v32; // rdi
  int v34; // eax
  int v35; // eax
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // r8
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  char v42; // cl
  unsigned __int64 v43; // r8
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rdx
  _QWORD *v46; // [rsp+0h] [rbp-A0h]
  _QWORD *v47; // [rsp+8h] [rbp-98h]
  unsigned __int64 v48; // [rsp+10h] [rbp-90h]
  _QWORD *v49; // [rsp+20h] [rbp-80h]
  unsigned __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned __int64 v51; // [rsp+28h] [rbp-78h]
  unsigned __int64 v52; // [rsp+30h] [rbp-70h]
  unsigned __int64 *v53; // [rsp+38h] [rbp-68h]
  unsigned __int64 v54; // [rsp+40h] [rbp-60h]
  _QWORD *v55; // [rsp+48h] [rbp-58h]
  _QWORD *v56; // [rsp+48h] [rbp-58h]
  __int64 v57; // [rsp+58h] [rbp-48h]
  unsigned __int64 v58; // [rsp+60h] [rbp-40h]
  unsigned __int64 v59; // [rsp+68h] [rbp-38h]
  _QWORD *v60; // [rsp+68h] [rbp-38h]

  v2 = a1;
  v3 = 0;
  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = a1 + 14;
  v58 = a2 >> 12;
  while ( 1 )
  {
    v8 = v4 >> 6;
    v9 = ~v7[v4 >> 6] & -(1LL << (v4 & 0x3F));
    if ( !v9 )
    {
      v10 = v8 + 1;
      if ( v8 == 7 )
        goto LABEL_19;
      while ( 1 )
      {
        v8 = v10;
        v9 = ~a1[v10 + 14];
        if ( a1[v10 + 14] != -1 )
          break;
        if ( ++v10 == 8 )
          goto LABEL_19;
      }
    }
    v11 = !_BitScanForward64(&v9, v9);
    if ( v11 )
      LODWORD(v9) = -1;
    v12 = (int)v9 + (v8 << 6);
    if ( v12 < 0x200 )
    {
      v13 = v12 >> 6;
      v14 = v7[v12 >> 6] & -(1LL << (v12 & 0x3F));
      if ( !v14 )
      {
        v15 = v13 + 1;
        if ( v13 == 7 )
        {
LABEL_23:
          v17 = 512;
LABEL_18:
          v5 = v17 - v12;
          v3 = v12;
          goto LABEL_19;
        }
        while ( 1 )
        {
          v14 = a1[v15 + 14];
          v13 = v15;
          if ( v14 )
            break;
          if ( ++v15 == 8 )
            goto LABEL_23;
        }
      }
      v11 = !_BitScanForward64((unsigned __int64 *)&v16, v14);
      if ( v11 )
        LODWORD(v16) = -1;
      v17 = (int)v16 + (v13 << 6);
      goto LABEL_18;
    }
LABEL_19:
    if ( a2 >> 12 <= v5 )
      break;
    v4 = v3 + v5;
    if ( v6 < v5 )
      v6 = v5;
  }
  v18 = v3;
  v51 = v3;
  v19 = v3 & 0x3F;
  v20 = v18 >> 6;
  v21 = &v7[v20];
  v57 = v20;
  v22 = v20 + 1;
  v23 = *v21;
  if ( v58 + v19 <= 0x40 )
  {
    v54 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v58) << v19;
    *v21 = v54 | v23;
    goto LABEL_38;
  }
  v59 = v58 + v19 - 64;
  v54 = 0xFFFFFFFFFFFFFFFFLL >> v19 << v19;
  *v21 = v54 | v23;
  if ( v59 > 0x40 )
  {
    v55 = v2;
    v24 = (v58 + v19 - 129) >> 6;
    memset(&v7[v20 + 1], 255, 8 * v24 + 8);
    v2 = v55;
    v25 = v59 - ((_BYTE)v24 << 6) - 64;
    v26 = v20 + v24 + 2;
    goto LABEL_28;
  }
  if ( v58 + v19 == 64 )
  {
LABEL_38:
    v2[13] += v58;
    v60 = v2;
    v53 = &v2[v57 + 23];
    v52 = *v53;
    v34 = sub_39FAC40(*v53 & v54);
    v28 = v60;
    v29 = v34;
LABEL_36:
    *v53 = v52 | v54;
  }
  else
  {
    v25 = v59;
    v26 = v20 + 1;
LABEL_28:
    v49 = v2;
    v7[v26] |= 0xFFFFFFFFFFFFFFFFLL >> (64 - v25);
    v2[13] += v58;
    v56 = v2 + 23;
    v53 = &v2[v57 + 23];
    v52 = *v53;
    v27 = sub_39FAC40(*v53 & v54);
    v28 = v49;
    v29 = v27;
    if ( v59 <= 0x40 )
    {
      if ( !v59 )
        goto LABEL_36;
      v35 = sub_39FAC40((0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v59)) & v56[v22]);
      v28 = v49;
      v29 += v35;
      *v53 = v54 | v52;
    }
    else
    {
      v46 = v49;
      v48 = v59 - 65;
      v50 = (v59 - 65) >> 6;
      v47 = v7;
      v30 = &v28[v57];
      v31 = (__int64)&v28[v20 + 1 + v50];
      do
      {
        v32 = v30[24];
        ++v30;
        v29 += (int)sub_39FAC40(v32);
      }
      while ( (_QWORD *)v31 != v30 );
      v7 = v47;
      v29 += (int)sub_39FAC40(
                    v56[v20 + 2 + v50]
                  & (0xFFFFFFFFFFFFFFFFLL >> (((_BYTE)v50 << 6) - ((unsigned __int8)v59 - 64) + 64)));
      *v53 = v54 | v52;
      memset(&v56[v57 + 1], 255, 8 * (v48 >> 6) + 8);
      v28 = v46;
      v59 = v59 - 64 - (v48 >> 6 << 6);
      v22 = v20 + (v48 >> 6) + 2;
    }
    v56[v22] |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v59);
  }
  v28[22] += v58 - v29;
  if ( v28[12] != v5 )
    return *v28 + (v51 << 12);
  v36 = v51 + v58;
  if ( v51 + v58 > 0x1FF )
    goto LABEL_61;
  while ( 2 )
  {
    v37 = v36 >> 6;
    v38 = ~v7[v36 >> 6] & -(1LL << (v36 & 0x3F));
    if ( !v38 )
    {
      v39 = v37 + 1;
      if ( v37 == 7 )
        goto LABEL_61;
      while ( 1 )
      {
        v37 = v39;
        v38 = ~v28[v39 + 14];
        if ( v28[v39 + 14] != -1 )
          break;
        if ( ++v39 == 8 )
          goto LABEL_61;
      }
    }
    v11 = !_BitScanForward64((unsigned __int64 *)&v40, v38);
    if ( v11 )
      LODWORD(v40) = -1;
    v41 = (int)v40 + (v37 << 6);
    v42 = v40 + ((_BYTE)v37 << 6);
    v43 = v41 >> 6;
    v44 = v7[v41 >> 6] & -(1LL << (v42 & 0x3F));
    if ( v44 )
    {
LABEL_53:
      v11 = !_BitScanForward64(&v44, v44);
      if ( v11 )
        LODWORD(v44) = -1;
      v36 = (v43 << 6) + (int)v44;
      if ( v36 - v41 == v5 )
        goto LABEL_60;
      if ( v6 < v36 - v41 )
        v6 = v36 - v41;
      continue;
    }
    break;
  }
  v45 = v43 + 1;
  if ( v43 != 7 )
  {
    do
    {
      v44 = v28[v45 + 14];
      v43 = v45;
      if ( v44 )
        goto LABEL_53;
    }
    while ( ++v45 != 8 );
  }
  if ( v5 == 512 - v41 )
  {
LABEL_60:
    v6 = v5;
    goto LABEL_61;
  }
  if ( v6 < 512 - v41 )
    v6 = 512 - v41;
LABEL_61:
  v28[12] = v6;
  return *v28 + (v51 << 12);
}
