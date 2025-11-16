// Function: sub_1342C60
// Address: 0x1342c60
//
_QWORD *__fastcall sub_1342C60(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, char a4, int a5)
{
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r13
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  char v14; // cl
  unsigned int v15; // eax
  unsigned __int64 v16; // rsi
  char v17; // cl
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  bool v21; // zf
  unsigned __int64 v22; // rsi
  unsigned int v23; // r14d
  unsigned __int64 v24; // r11
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r10
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  int v29; // eax
  unsigned __int64 v30; // r14
  char v31; // cl
  unsigned __int64 v32; // r14
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rcx
  bool v36; // al
  __int64 v38; // rsi
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  unsigned int v41; // eax
  char v42; // cl
  unsigned int v43; // eax
  unsigned int v44; // r14d
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  char v48; // cl
  unsigned int v49; // eax
  unsigned __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rax
  unsigned int v53; // ecx
  unsigned __int64 v54; // r13
  char v55; // r15
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rcx
  unsigned int v66; // [rsp+18h] [rbp-38h]

  v5 = (a3 + 4095) & 0xFFFFFFFFFFFFF000LL;
  v6 = v5 + a2 - 4096;
  if ( a2 > v6 )
    return 0;
  v10 = sub_130FA30(v5 + a2 - 4096);
  if ( v10 > 0x7000000000000000LL )
  {
    v16 = 199;
    v17 = -57;
  }
  else
  {
    v11 = v10 - 1;
    _BitScanReverse64(&v12, v10);
    v13 = v12 - ((((v10 - 1) & v10) == 0) - 1);
    if ( v13 < 0xE )
      v13 = 14;
    v14 = v13 - 3;
    v15 = v13 - 14;
    if ( !v15 )
      v14 = 12;
    v16 = ((v11 >> v14) & 3) + 4 * v15;
    v17 = v16;
  }
  if ( !a4 )
  {
    v18 = v16 >> 6;
    v19 = *(_QWORD *)(a1 + 8 * v18) & -(1LL << v17);
    if ( !v19 )
    {
      v20 = v18 + 1;
      if ( v18 == 3 )
        goto LABEL_37;
      while ( 1 )
      {
        v19 = *(_QWORD *)(a1 + 8 * v20);
        v18 = v20;
        if ( v19 )
          break;
        if ( ++v20 == 4 )
          goto LABEL_37;
      }
    }
    v21 = !_BitScanForward64(&v19, v19);
    if ( v21 )
      LODWORD(v19) = -1;
    v22 = (int)v19 + (v18 << 6);
    v23 = v22;
    if ( v22 <= 0xC7 )
    {
      v24 = 0;
      v25 = 0;
      v26 = 0;
      do
      {
        if ( a5 == 64 )
          a5 = 63;
        if ( v6 < qword_5060180[v23] >> a5 )
          break;
        if ( !v26 )
          goto LABEL_74;
        v27 = a1 + 32LL * v23;
        v28 = *(_QWORD *)(v27 + 56);
        v29 = (*(_QWORD *)(v27 + 48) > v25) - (*(_QWORD *)(v27 + 48) < v25);
        if ( !v29 )
          v29 = (v28 > v24) - (v28 < v24);
        if ( v29 == -1 )
        {
LABEL_74:
          v62 = sub_133F530((_QWORD *)(a1 + 32LL * (v23 + 1)));
          v63 = 32LL * (v23 + 1);
          v26 = v62;
          v25 = *(_QWORD *)(a1 + v63 + 16);
          v24 = *(_QWORD *)(a1 + v63 + 24);
        }
        if ( v23 == 199 )
          break;
        v30 = v23 + 1;
        v31 = v30;
        v32 = v30 >> 6;
        v33 = *(_QWORD *)(a1 + 8 * v32) & -(1LL << v31);
        if ( !v33 )
        {
          v34 = v32 + 1;
          if ( v32 == 3 )
            break;
          while ( 1 )
          {
            v33 = *(_QWORD *)(a1 + 8 * v34);
            v32 = v34;
            if ( v33 )
              break;
            if ( ++v34 == 4 )
              goto LABEL_34;
          }
        }
        v21 = !_BitScanForward64(&v33, v33);
        if ( v21 )
          LODWORD(v33) = -1;
        v35 = (v32 << 6) + (int)v33;
        v23 = v35;
      }
      while ( v35 <= 0xC7 );
LABEL_34:
      v36 = v26 == 0;
      if ( a3 <= 0x1000 )
        return v26;
      goto LABEL_39;
    }
    goto LABEL_37;
  }
  v38 = 32 * (v16 + 1);
  if ( sub_133F520((_QWORD *)(a1 + v38)) )
  {
LABEL_37:
    v36 = 1;
    v26 = 0;
    goto LABEL_38;
  }
  v26 = sub_133F530((_QWORD *)(a1 + v38));
  v36 = v26 == 0;
LABEL_38:
  if ( a3 <= 0x1000 )
    return v26;
LABEL_39:
  if ( v36 )
  {
    v39 = sub_130FA30(a2);
    if ( v39 > 0x7000000000000000LL )
    {
      v44 = 199;
    }
    else
    {
      _BitScanReverse64(&v40, v39);
      v41 = v40 - ((((v39 - 1) & v39) == 0) - 1);
      if ( v41 < 0xE )
        v41 = 14;
      v42 = v41 - 3;
      v43 = v41 - 14;
      if ( !v43 )
        v42 = 12;
      v44 = (((v39 - 1) >> v42) & 3) + 4 * v43;
    }
    v45 = sub_130FA30(v6);
    if ( v45 > 0x7000000000000000LL )
    {
      v66 = 199;
    }
    else
    {
      _BitScanReverse64(&v46, v45);
      v47 = v46 - ((((v45 - 1) & v45) == 0) - 1);
      if ( v47 < 0xE )
        v47 = 14;
      v48 = v47 - 3;
      v49 = v47 - 14;
      if ( !v49 )
        v48 = 12;
      v66 = (((v45 - 1) >> v48) & 3) + 4 * v49;
    }
    v50 = (unsigned __int64)v44 >> 6;
    v51 = *(_QWORD *)(a1 + 8 * v50) & -(1LL << v44);
    if ( !v51 )
    {
      v52 = v50 + 1;
      if ( v50 == 3 )
      {
LABEL_77:
        v53 = 200;
        goto LABEL_60;
      }
      while ( 1 )
      {
        v51 = *(_QWORD *)(a1 + 8 * v52);
        LODWORD(v50) = v52;
        if ( v51 )
          break;
        if ( ++v52 == 4 )
          goto LABEL_77;
      }
    }
    v21 = !_BitScanForward64(&v51, v51);
    if ( v21 )
      LODWORD(v51) = -1;
    v53 = v51 + ((_DWORD)v50 << 6);
LABEL_60:
    if ( v66 > v53 )
    {
      while ( 1 )
      {
        v54 = v53 + 1;
        v55 = v53 + 1;
        v26 = sub_133F530((_QWORD *)(a1 + 32 * v54));
        v56 = v26[1] & 0xFFFFFFFFFFFFF000LL;
        v57 = -(__int64)v5 & (v56 + v5 - 1);
        if ( v56 <= v57 )
        {
          v58 = (v26[2] & 0xFFFFFFFFFFFFF000LL) + v56;
          if ( v57 < v58 && a2 <= v58 - v57 )
            return v26;
        }
        v59 = v54 >> 6;
        v60 = *(_QWORD *)(a1 + 8 * (v54 >> 6)) & -(1LL << v55);
        if ( v60 )
          goto LABEL_69;
        v61 = v59 + 1;
        if ( v59 != 3 )
          break;
LABEL_76:
        v53 = 200;
LABEL_72:
        if ( v66 <= v53 )
          return 0;
      }
      while ( 1 )
      {
        v60 = *(_QWORD *)(a1 + 8 * v61);
        LODWORD(v59) = v61;
        if ( v60 )
          break;
        if ( ++v61 == 4 )
          goto LABEL_76;
      }
LABEL_69:
      v21 = !_BitScanForward64(&v60, v60);
      if ( v21 )
        LODWORD(v60) = -1;
      v53 = v60 + ((_DWORD)v59 << 6);
      goto LABEL_72;
    }
    return 0;
  }
  return v26;
}
