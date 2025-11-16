// Function: sub_11EE290
// Address: 0x11ee290
//
__int64 __fastcall sub_11EE290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v8; // rax
  char v9; // al
  _QWORD *v10; // r9
  char v11; // al
  __int64 v12; // rsi
  __int64 *v13; // rbx
  char *v14; // rax
  size_t v15; // rdx
  __int64 v16; // rdx
  int v17; // esi
  int v18; // ecx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rbx
  unsigned __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned int *v30; // rax
  __int64 v31; // rbx
  int v32; // eax
  __int64 v33; // rdi
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // r12
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r13
  unsigned __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // ecx
  char v49; // al
  __int64 v50; // rsi
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rax
  _QWORD *v53; // rcx
  _QWORD *v54; // rcx
  __int64 *v55; // [rsp+8h] [rbp-C8h]
  int v56; // [rsp+14h] [rbp-BCh]
  unsigned int v57; // [rsp+20h] [rbp-B0h] BYREF
  int v58; // [rsp+24h] [rbp-ACh] BYREF
  __int64 v59; // [rsp+28h] [rbp-A8h]
  _QWORD v60[4]; // [rsp+30h] [rbp-A0h] BYREF
  char v61; // [rsp+50h] [rbp-80h]
  char v62; // [rsp+51h] [rbp-7Fh]
  __int64 v63; // [rsp+60h] [rbp-70h]
  _QWORD v64[2]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v65; // [rsp+78h] [rbp-58h]
  __int64 v66; // [rsp+80h] [rbp-50h]
  __int16 v67; // [rsp+88h] [rbp-48h]
  __int64 v68[8]; // [rsp+90h] [rbp-40h] BYREF

  if ( !sub_B451B0(a2) )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v5 = 0;
    }
  }
  v6 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v6 != 85 )
    return 0;
  if ( !sub_B451B0(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) )
    return 0;
  v8 = *(_QWORD *)(v6 + 16);
  if ( !v8 || *(_QWORD *)(v8 + 8) )
    return 0;
  v57 = 524;
  v56 = sub_B49240(v6);
  v55 = *(__int64 **)(a1 + 24);
  v9 = sub_A73ED0((_QWORD *)(v6 + 72), 23);
  v10 = (_QWORD *)(v6 + 72);
  if ( !v9 && (v11 = sub_B49560(v6, 23), v10 = (_QWORD *)(v6 + 72), !v11)
    || (unsigned __int8)sub_A73ED0(v10, 4)
    || (unsigned __int8)sub_B49560(v6, 4) )
  {
    v12 = *(_QWORD *)(v6 - 32);
    if ( v12 )
    {
      if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == *(_QWORD *)(v6 + 80) )
        sub_981210(*v55, v12, &v57);
    }
  }
  v13 = *(__int64 **)(a1 + 24);
  v14 = (char *)sub_BD5D20(v5);
  if ( !(unsigned __int8)sub_980AF0(*v13, v14, v15, &v58) )
  {
    if ( *(_DWORD *)(v5 + 36) != 335 )
      return 0;
    v47 = *(_QWORD *)(a2 + 8);
    v48 = *(unsigned __int8 *)(v47 + 8);
    v49 = *(_BYTE *)(v47 + 8);
    if ( (unsigned int)(v48 - 17) > 1 )
    {
      if ( (_BYTE)v48 != 2 )
      {
LABEL_83:
        if ( v48 != 17 )
        {
LABEL_66:
          if ( v49 == 3 )
          {
            v19 = 228;
            v18 = 231;
            v17 = 227;
            goto LABEL_22;
          }
          return 0;
        }
        v50 = *(_QWORD *)(v47 + 16);
LABEL_65:
        v49 = *(_BYTE *)(*(_QWORD *)v50 + 8LL);
        goto LABEL_66;
      }
    }
    else
    {
      v50 = *(_QWORD *)(v47 + 16);
      if ( *(_BYTE *)(*(_QWORD *)v50 + 8LL) != 2 )
      {
        if ( v48 == 18 )
          goto LABEL_65;
        goto LABEL_83;
      }
    }
    v19 = 229;
    v18 = 232;
    v17 = 234;
    goto LABEL_22;
  }
  v16 = (unsigned int)(v58 - 448);
  if ( (unsigned int)v16 > 2 )
    return 0;
  v17 = dword_3F94F00[v16];
  v18 = v58 - 217;
  v19 = v58 - 220;
LABEL_22:
  if ( v57 != v18 && v57 != v17 && v57 != v19 && (v56 & 0xFFFFFFFD) != 0x58 )
    return 0;
  v20 = *(_QWORD *)(a3 + 48);
  v63 = a3;
  v64[0] = 0;
  v64[1] = 0;
  v65 = v20;
  if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
    sub_BD73F0((__int64)v64);
  v21 = *(_QWORD *)(a3 + 56);
  v67 = *(_WORD *)(a3 + 64);
  v66 = v21;
  sub_B33910(v68, (__int64 *)a3);
  v22 = *(_QWORD *)(v6 + 40);
  *(_WORD *)(a3 + 64) = 0;
  *(_QWORD *)(a3 + 48) = v22;
  *(_QWORD *)(a3 + 56) = v6 + 24;
  v23 = *(_QWORD *)sub_B46C60(v6);
  v60[0] = v23;
  if ( !v23 || (sub_B96E90((__int64)v60, v23, 1), (v26 = v60[0]) == 0) )
  {
    sub_93FB40(a3, 0);
    v26 = v60[0];
    goto LABEL_80;
  }
  v27 = *(unsigned int *)(a3 + 8);
  v28 = *(_QWORD *)a3;
  v29 = *(_QWORD *)a3 + 16 * v27;
  if ( *(_QWORD *)a3 == v29 )
  {
LABEL_76:
    v52 = *(unsigned int *)(a3 + 12);
    if ( v27 >= v52 )
    {
      if ( v52 < v27 + 1 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v27 + 1, 0x10u, v24, v25);
        v28 = *(_QWORD *)a3;
        v27 = *(unsigned int *)(a3 + 8);
      }
      v53 = (_QWORD *)(16 * v27 + v28);
      *v53 = 0;
      v53[1] = v26;
      v26 = v60[0];
      ++*(_DWORD *)(a3 + 8);
    }
    else
    {
      if ( v29 )
      {
        *(_DWORD *)v29 = 0;
        *(_QWORD *)(v29 + 8) = v26;
        v26 = v60[0];
      }
      ++*(_DWORD *)(a3 + 8);
    }
LABEL_80:
    if ( !v26 )
      goto LABEL_36;
    goto LABEL_35;
  }
  v30 = *(unsigned int **)a3;
  while ( 1 )
  {
    v25 = *v30;
    if ( !(_DWORD)v25 )
      break;
    v30 += 4;
    if ( (unsigned int *)v29 == v30 )
      goto LABEL_76;
  }
  *((_QWORD *)v30 + 1) = v60[0];
LABEL_35:
  sub_B91220((__int64)v60, v26);
LABEL_36:
  v31 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  v62 = 1;
  v60[0] = "merged.sqrt";
  v61 = 3;
  v32 = sub_B45210(a2);
  v33 = *(_QWORD *)(v31 + 8);
  BYTE4(v59) = 1;
  LODWORD(v59) = v32;
  v34 = sub_AD8DD0(v33, 0.5);
  v35 = sub_A826E0((unsigned int **)a3, (_BYTE *)v31, v34, v59, (__int64)v60, 0);
  v36 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v36 )
  {
    v37 = *(_QWORD *)(v36 + 8);
    **(_QWORD **)(v36 + 16) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = *(_QWORD *)(v36 + 16);
  }
  *(_QWORD *)v36 = v35;
  if ( v35 )
  {
    v38 = *(_QWORD *)(v35 + 16);
    *(_QWORD *)(v36 + 8) = v38;
    if ( v38 )
      *(_QWORD *)(v38 + 16) = v36 + 8;
    *(_QWORD *)(v36 + 16) = v35 + 16;
    *(_QWORD *)(v35 + 16) = v36;
  }
  v39 = v63;
  if ( v65 )
  {
    sub_A88F30(v63, v65, v66, v67);
    v39 = v63;
  }
  else
  {
    *(_QWORD *)(v63 + 48) = 0;
    *(_QWORD *)(v39 + 56) = 0;
    *(_WORD *)(v39 + 64) = 0;
  }
  v60[0] = v68[0];
  if ( !v68[0] || (sub_B96E90((__int64)v60, v68[0], 1), (v42 = v60[0]) == 0) )
  {
    sub_93FB40(v39, 0);
    goto LABEL_72;
  }
  v43 = *(unsigned int *)(v39 + 8);
  v44 = *(_QWORD *)v39;
  v45 = *(_QWORD *)v39 + 16 * v43;
  if ( *(_QWORD *)v39 == v45 )
  {
LABEL_68:
    v51 = *(unsigned int *)(v39 + 12);
    if ( v43 >= v51 )
    {
      if ( v51 < v43 + 1 )
      {
        sub_C8D5F0(v39, (const void *)(v39 + 16), v43 + 1, 0x10u, v40, v41);
        v44 = *(_QWORD *)v39;
        v43 = *(unsigned int *)(v39 + 8);
      }
      v54 = (_QWORD *)(16 * v43 + v44);
      *v54 = 0;
      v54[1] = v42;
      ++*(_DWORD *)(v39 + 8);
    }
    else
    {
      if ( v45 )
      {
        *(_DWORD *)v45 = 0;
        *(_QWORD *)(v45 + 8) = v42;
      }
      ++*(_DWORD *)(v39 + 8);
    }
LABEL_72:
    v42 = v60[0];
    if ( !v60[0] )
      goto LABEL_53;
    goto LABEL_52;
  }
  v46 = *(_QWORD *)v39;
  while ( *(_DWORD *)v46 )
  {
    v46 += 16;
    if ( v45 == v46 )
      goto LABEL_68;
  }
  *(_QWORD *)(v46 + 8) = v60[0];
LABEL_52:
  sub_B91220((__int64)v60, v42);
LABEL_53:
  if ( v68[0] )
    sub_B91220((__int64)v68, v68[0]);
  if ( v65 != 0 && v65 != -4096 && v65 != -8192 )
    sub_BD60C0(v64);
  return v6;
}
