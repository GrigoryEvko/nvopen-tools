// Function: sub_2B22310
// Address: 0x2b22310
//
bool __fastcall sub_2B22310(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  unsigned __int8 *v6; // r12
  char v7; // cl
  __int64 v8; // rsi
  int i; // eax
  unsigned int v10; // edx
  __int64 v11; // rdi
  unsigned __int8 *v12; // r8
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 *v15; // rbx
  __int64 v16; // rsi
  __int64 *v17; // r15
  _BYTE **v18; // r8
  int v19; // eax
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 *v27; // rdi
  __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 *v30; // r9
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 *v35; // r11
  unsigned int v36; // eax
  __int64 *v37; // r9
  __int64 v38; // r10
  __int64 v39; // rcx
  unsigned int v40; // eax
  __int64 *v41; // r9
  __int64 v42; // r10
  __int64 v43; // rcx
  unsigned int v44; // eax
  __int64 *v45; // r9
  __int64 v46; // r10
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  int v50; // r10d
  int v51; // r9d
  int v52; // r9d
  int v53; // r9d
  __int64 v54; // rdi
  int v55; // r9d
  int v56; // r11d
  int v57; // edi
  __int64 v58; // rdi
  unsigned int v59; // r8d
  __int64 *v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rcx
  unsigned int v63; // r8d
  __int64 *v64; // rax
  __int64 v65; // rsi
  unsigned int v66; // eax
  __int64 *v67; // rsi
  __int64 v68; // rdi
  int v69; // r9d
  int v70; // eax
  int v71; // esi
  int v72; // esi
  int v73; // r8d
  int v74; // eax
  int v75; // edi
  int v76; // [rsp+Ch] [rbp-84h]
  int v77; // [rsp+Ch] [rbp-84h]
  int v78; // [rsp+Ch] [rbp-84h]
  unsigned int v79; // [rsp+18h] [rbp-78h]
  __int64 *v82; // [rsp+30h] [rbp-60h]
  unsigned int v83; // [rsp+38h] [rbp-58h]
  __int64 v84; // [rsp+48h] [rbp-48h] BYREF
  __int64 v85; // [rsp+50h] [rbp-40h] BYREF
  __int64 v86; // [rsp+58h] [rbp-38h]

  v4 = a1;
  if ( a2 != a1 )
  {
    v6 = *(unsigned __int8 **)(a1 + 24);
    v7 = *(_BYTE *)(a3 + 88) & 1;
    if ( !v7 )
      goto LABEL_26;
LABEL_3:
    v8 = a3 + 96;
    for ( i = 3; ; i = v33 - 1 )
    {
      v10 = i & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v11 = v8 + 72LL * v10;
      v12 = *(unsigned __int8 **)v11;
      if ( v6 != *(unsigned __int8 **)v11 )
      {
        v57 = 1;
        while ( v12 != (unsigned __int8 *)-4096LL )
        {
          v69 = v57 + 1;
          v10 = i & (v57 + v10);
          v11 = v8 + 72LL * v10;
          v12 = *(unsigned __int8 **)v11;
          if ( v6 == *(unsigned __int8 **)v11 )
            goto LABEL_5;
          v57 = v69;
        }
        if ( !v7 )
        {
          v33 = *(unsigned int *)(a3 + 104);
LABEL_56:
          v54 = 72 * v33;
          goto LABEL_57;
        }
        v54 = 288;
LABEL_57:
        v11 = v8 + v54;
      }
LABEL_5:
      v13 = 288;
      if ( !v7 )
        v13 = 72LL * *(unsigned int *)(a3 + 104);
      if ( v11 != v8 + v13 )
      {
        v14 = *(unsigned int *)(v11 + 16);
        v15 = *(__int64 **)(v11 + 8);
        v84 = *(_QWORD *)(a4 + 184);
        if ( v14 )
        {
          v16 = 8 * v14;
          v17 = &v15[v14];
          if ( v17 == sub_2B0BA00(v15, (__int64)v17, &v84) )
            break;
        }
      }
LABEL_24:
      v4 = *(_QWORD *)(v4 + 8);
      if ( a2 == v4 )
        return a2 == v4;
      v6 = *(unsigned __int8 **)(v4 + 24);
      v7 = *(_BYTE *)(a3 + 88) & 1;
      if ( v7 )
        goto LABEL_3;
LABEL_26:
      v33 = *(unsigned int *)(a3 + 104);
      v8 = *(_QWORD *)(a3 + 96);
      if ( !(_DWORD)v33 )
        goto LABEL_56;
    }
    v19 = *v6;
    if ( (unsigned __int8)v19 <= 0x1Cu )
      return a2 == v4;
    v21 = (unsigned int)(v19 - 42);
    if ( (unsigned __int8)v21 > 0x36u )
      return a2 == v4;
    v22 = 0x40143FFE03FFFFLL;
    if ( !_bittest64(&v22, v21)
      || (unsigned __int8)(*v18[52] - 42) > 0x36u
      || !_bittest64(&v22, (unsigned int)(unsigned __int8)*v18[52] - 42) )
    {
      return a2 == v4;
    }
    v85 = sub_9208B0(*(_QWORD *)(a3 + 3344), *(_QWORD *)(*(_QWORD *)*v18 + 8LL));
    v86 = v23;
    v79 = sub_CA1930(&v85);
    v24 = v16 >> 3;
    if ( v16 >> 5 )
    {
      v25 = *(_QWORD *)(a3 + 3528);
      v83 = *(_DWORD *)(a3 + 3544);
      v82 = &v15[4 * (v16 >> 5)];
      v26 = v83 - 1;
      v27 = (__int64 *)(v25 + 24LL * v83);
      while ( 1 )
      {
        v28 = *v15;
        if ( !v83 )
          goto LABEL_22;
        v29 = v26 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v30 = (__int64 *)(v25 + 24LL * v29);
        v31 = *v30;
        if ( *v30 != v28 )
          break;
LABEL_20:
        if ( v27 == v30 || v79 >= (unsigned __int64)v30[1] )
          goto LABEL_22;
        v34 = v15[1];
        v35 = v15 + 1;
        v36 = v26 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v37 = (__int64 *)(v25 + 24LL * v36);
        v38 = *v37;
        if ( *v37 != v34 )
        {
          v51 = 1;
          while ( v38 != -4096 )
          {
            v36 = v26 & (v51 + v36);
            v78 = v51 + 1;
            v37 = (__int64 *)(v25 + 24LL * v36);
            v38 = *v37;
            if ( v34 == *v37 )
              goto LABEL_29;
            v51 = v78;
          }
          goto LABEL_46;
        }
LABEL_29:
        if ( v27 == v37 || v79 >= (unsigned __int64)v37[1] )
          goto LABEL_46;
        v39 = v15[2];
        v35 = v15 + 2;
        v40 = v26 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v41 = (__int64 *)(v25 + 24LL * v40);
        v42 = *v41;
        if ( v39 != *v41 )
        {
          v52 = 1;
          while ( v42 != -4096 )
          {
            v40 = v26 & (v52 + v40);
            v76 = v52 + 1;
            v41 = (__int64 *)(v25 + 24LL * v40);
            v42 = *v41;
            if ( v39 == *v41 )
              goto LABEL_32;
            v52 = v76;
          }
          goto LABEL_46;
        }
LABEL_32:
        if ( v27 == v41 || v79 >= (unsigned __int64)v41[1] )
          goto LABEL_46;
        v43 = v15[3];
        v35 = v15 + 3;
        v44 = v26 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v45 = (__int64 *)(v25 + 24LL * v44);
        v46 = *v45;
        if ( v43 != *v45 )
        {
          v53 = 1;
          while ( v46 != -4096 )
          {
            v44 = v26 & (v53 + v44);
            v77 = v53 + 1;
            v45 = (__int64 *)(v25 + 24LL * v44);
            v46 = *v45;
            if ( v43 == *v45 )
              goto LABEL_35;
            v53 = v77;
          }
LABEL_46:
          v15 = v35;
          goto LABEL_22;
        }
LABEL_35:
        if ( v27 == v45 || v79 >= (unsigned __int64)v45[1] )
          goto LABEL_46;
        v15 += 4;
        if ( v82 == v15 )
        {
          v24 = v17 - v15;
          goto LABEL_39;
        }
      }
      v55 = 1;
      while ( v31 != -4096 )
      {
        v56 = v55 + 1;
        v29 = v26 & (v55 + v29);
        v30 = (__int64 *)(v25 + 24LL * v29);
        v31 = *v30;
        if ( v28 == *v30 )
          goto LABEL_20;
        v55 = v56;
      }
      goto LABEL_22;
    }
LABEL_39:
    switch ( v24 )
    {
      case 2LL:
        v47 = *(unsigned int *)(a3 + 3544);
        v48 = *(_QWORD *)(a3 + 3528);
        v62 = *v15;
        if ( !(_DWORD)v47 )
          goto LABEL_22;
        v50 = v47 - 1;
        break;
      case 3LL:
        v47 = *(unsigned int *)(a3 + 3544);
        v58 = *v15;
        v48 = *(_QWORD *)(a3 + 3528);
        if ( !(_DWORD)v47 )
          goto LABEL_22;
        v50 = v47 - 1;
        v59 = (v47 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v60 = (__int64 *)(v48 + 24LL * v59);
        v61 = *v60;
        if ( v58 != *v60 )
        {
          v70 = 1;
          while ( v61 != -4096 )
          {
            v71 = v70 + 1;
            v59 = v50 & (v70 + v59);
            v60 = (__int64 *)(v48 + 24LL * v59);
            v61 = *v60;
            if ( v58 == *v60 )
              goto LABEL_70;
            v70 = v71;
          }
          goto LABEL_22;
        }
LABEL_70:
        if ( v60 == (__int64 *)(v48 + 24LL * (unsigned int)v47) || v60[1] <= (unsigned __int64)v79 )
          goto LABEL_22;
        v62 = v15[1];
        ++v15;
        break;
      case 1LL:
        v47 = *(unsigned int *)(a3 + 3544);
        v48 = *(_QWORD *)(a3 + 3528);
        v49 = *v15;
        if ( !(_DWORD)v47 )
        {
LABEL_22:
          if ( v17 == v15 )
            return a2 == v4;
          v85 = sub_9208B0(*(_QWORD *)(a3 + 3344), *((_QWORD *)v6 + 1));
          v86 = v32;
          if ( sub_CA1930(&v85) > (unsigned __int64)v79 )
            return a2 == v4;
          goto LABEL_24;
        }
        v50 = v47 - 1;
LABEL_77:
        v66 = v50 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v67 = (__int64 *)(v48 + 24LL * v66);
        v68 = *v67;
        if ( *v67 == v49 )
        {
LABEL_78:
          if ( v67 != (__int64 *)(v48 + 24 * v47) && v79 < (unsigned __int64)v67[1] )
            return a2 == v4;
        }
        else
        {
          v72 = 1;
          while ( v68 != -4096 )
          {
            v73 = v72 + 1;
            v66 = v50 & (v66 + v72);
            v67 = (__int64 *)(v48 + 24LL * v66);
            v68 = *v67;
            if ( *v67 == v49 )
              goto LABEL_78;
            v72 = v73;
          }
        }
        goto LABEL_22;
      default:
        return a2 == v4;
    }
    v63 = v50 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
    v64 = (__int64 *)(v48 + 24LL * v63);
    v65 = *v64;
    if ( v62 != *v64 )
    {
      v74 = 1;
      while ( v65 != -4096 )
      {
        v75 = v74 + 1;
        v63 = v50 & (v74 + v63);
        v64 = (__int64 *)(v48 + 24LL * v63);
        v65 = *v64;
        if ( v62 == *v64 )
          goto LABEL_74;
        v74 = v75;
      }
      goto LABEL_22;
    }
LABEL_74:
    if ( v64 == (__int64 *)(v48 + 24LL * (unsigned int)v47) || v64[1] <= (unsigned __int64)v79 )
      goto LABEL_22;
    v49 = v15[1];
    ++v15;
    goto LABEL_77;
  }
  return a2 == v4;
}
