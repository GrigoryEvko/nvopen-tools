// Function: sub_1482AA0
// Address: 0x1482aa0
//
__int64 __fastcall sub_1482AA0(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rax
  _QWORD *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // r14
  int v23; // edi
  __int64 v24; // rbx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // eax
  int v29; // r15d
  __int64 v30; // r10
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r11
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // r8
  int v40; // eax
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 *v45; // rdx
  __int64 v46; // r8
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 *v51; // rcx
  char v52; // al
  __int64 *v53; // rcx
  __int64 *v54; // r9
  __int64 *v55; // rdx
  char v56; // al
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rax
  int v61; // eax
  int v62; // edx
  int v63; // r9d
  int v64; // edx
  int v65; // r11d
  int v66; // r8d
  __int64 *v67; // [rsp-70h] [rbp-70h]
  __int64 *v68; // [rsp-70h] [rbp-70h]
  __int64 *v69; // [rsp-68h] [rbp-68h]
  __int64 *v70; // [rsp-68h] [rbp-68h]
  __int64 *v71; // [rsp-68h] [rbp-68h]
  __int64 v72; // [rsp-68h] [rbp-68h]
  __int64 v73; // [rsp-68h] [rbp-68h]
  int v74; // [rsp-68h] [rbp-68h]
  __int64 v75; // [rsp-60h] [rbp-60h]
  _QWORD v76[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v77[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v8 = a2 - 48;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(_QWORD *)(a2 - 8);
  v11 = (_QWORD *)(v8 + 24LL * *(unsigned int *)(a2 + 56) + 8);
  v12 = v8 + 24LL * *(unsigned int *)(a2 + 56) + 24;
  if ( sub_15CC510(a1[7], *v11, 3LL * *(unsigned int *)(a2 + 56), a6, a7) )
  {
    v13 = a1[7];
    v14 = *(unsigned int *)(v13 + 48);
    if ( (_DWORD)v14 )
    {
      v15 = v11[1];
      v16 = *(_QWORD *)(v13 + 32);
      v17 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v15 == *v18 )
      {
LABEL_8:
        if ( v18 != (__int64 *)(v16 + 16 * v14) && v18[1] )
          goto LABEL_12;
      }
      else
      {
        v64 = 1;
        while ( v19 != -8 )
        {
          v65 = v64 + 1;
          v17 = (v14 - 1) & (v17 + v64);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v15 == *v18 )
            goto LABEL_8;
          v64 = v65;
        }
      }
    }
    ++v11;
  }
  if ( v11 != (_QWORD *)v12 )
    return 0;
LABEL_12:
  v20 = a1[8];
  v21 = *(_QWORD *)(a2 + 40);
  v22 = 0;
  v23 = *(_DWORD *)(v20 + 24);
  v24 = *(_QWORD *)(v20 + 8);
  v75 = v21;
  if ( v23 )
  {
    v25 = (v23 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( *v26 == v21 )
    {
LABEL_14:
      v22 = v26[1];
    }
    else
    {
      v61 = 1;
      while ( v27 != -8 )
      {
        v66 = v61 + 1;
        v25 = (v23 - 1) & (v61 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( v21 == *v26 )
          goto LABEL_14;
        v61 = v66;
      }
      v22 = 0;
    }
  }
  v28 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v28 )
  {
    v29 = v23 - 1;
    v30 = a2 - 24LL * v28;
    v31 = 24LL * *(unsigned int *)(a2 + 56);
    v32 = v31 + 8;
    v33 = v31 + 8LL * (v28 - 1) + 16;
    while ( 1 )
    {
      v35 = v30;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v35 = *(_QWORD *)(a2 - 8);
      v34 = 0;
      if ( !v23 )
        goto LABEL_18;
      v36 = *(_QWORD *)(v35 + v32);
      v37 = v29 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v38 = (__int64 *)(v24 + 16LL * v37);
      v39 = *v38;
      if ( *v38 == v36 )
        break;
      v40 = 1;
      while ( v39 != -8 )
      {
        v37 = v29 & (v40 + v37);
        v74 = v40 + 1;
        v38 = (__int64 *)(v24 + 16LL * v37);
        v39 = *v38;
        if ( v36 == *v38 )
          goto LABEL_17;
        v40 = v74;
      }
      if ( v22 )
        return 0;
LABEL_19:
      v32 += 8;
      if ( v32 == v33 )
        goto LABEL_28;
    }
LABEL_17:
    v34 = v38[1];
LABEL_18:
    if ( v22 != v34 )
      return 0;
    goto LABEL_19;
  }
LABEL_28:
  v41 = a1[7];
  v42 = *(unsigned int *)(v41 + 48);
  if ( !(_DWORD)v42 )
LABEL_61:
    BUG();
  v43 = *(_QWORD *)(v41 + 32);
  v44 = ((_DWORD)v42 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
  v45 = (__int64 *)(v43 + 16 * v44);
  v46 = *v45;
  if ( v75 != *v45 )
  {
    v62 = 1;
    while ( v46 != -8 )
    {
      v63 = v62 + 1;
      LODWORD(v44) = (v42 - 1) & (v62 + v44);
      v45 = (__int64 *)(v43 + 16LL * (unsigned int)v44);
      v46 = *v45;
      if ( v75 == *v45 )
        goto LABEL_30;
      v62 = v63;
    }
    goto LABEL_61;
  }
LABEL_30:
  if ( v45 == (__int64 *)(v43 + 16 * v42) )
    goto LABEL_61;
  v47 = sub_157EBA0(**(_QWORD **)(v45[1] + 8));
  if ( *(_BYTE *)(v47 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v47 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v48 = *(_QWORD *)(v47 - 24);
  v49 = *(_QWORD *)(v47 - 72);
  v76[0] = *(_QWORD *)(v47 + 40);
  v76[1] = v48;
  v50 = *(_QWORD *)(v47 - 48);
  v77[0] = v76[0];
  v77[1] = v50;
  if ( !(unsigned __int8)sub_15CC350(v76) )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v51 = *(__int64 **)(a2 - 8);
  else
    v51 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v67 = v51 + 3;
  v69 = v51;
  v52 = sub_15CCFD0(v41, v76, v51);
  v53 = v69;
  v54 = v67;
  if ( v52 && (v55 = v67, v68 = v69, v70 = v54, v56 = sub_15CCFD0(v41, v77, v55), v54 = v70, v53 = v68, v56) )
  {
    v57 = *v68;
    v58 = v68[3];
  }
  else
  {
    v71 = v53;
    if ( !(unsigned __int8)sub_15CCFD0(v41, v76, v54) || !(unsigned __int8)sub_15CCFD0(v41, v77, v71) )
      return 0;
    v57 = v71[3];
    v58 = *v71;
  }
  v72 = *(_QWORD *)(a2 + 40);
  v59 = sub_146F1B0((__int64)a1, v57);
  if ( !(unsigned __int8)sub_145B840(v22, a1[7], v59, v72) )
    return 0;
  v73 = *(_QWORD *)(a2 + 40);
  v60 = sub_146F1B0((__int64)a1, v58);
  if ( !(unsigned __int8)sub_145B840(v22, a1[7], v60, v73) )
    return 0;
  return sub_1482570(a1, (__int64 *)a2, v49, v57, v58, a3, a4);
}
