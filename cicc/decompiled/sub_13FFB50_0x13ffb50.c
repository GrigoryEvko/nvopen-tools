// Function: sub_13FFB50
// Address: 0x13ffb50
//
__int64 *__fastcall sub_13FFB50(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v3; // rax
  __int64 *v5; // rdx
  __int64 v6; // rcx
  __int64 *i; // rax
  unsigned int v8; // esi
  __int64 v9; // rdi
  int v10; // r11d
  __int64 *v11; // r10
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r14
  unsigned int v16; // ebx
  __int64 v17; // r15
  __int64 v18; // rax
  int v19; // esi
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // edi
  __int64 *v23; // rdx
  __int64 v24; // r10
  __int64 *v25; // rdx
  __int64 *v26; // rax
  __int64 *v27; // rsi
  __int64 **v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // r9
  unsigned int v36; // r8d
  __int64 v37; // r11
  unsigned int v38; // r9d
  __int64 **v39; // rdi
  __int64 *v40; // r10
  __int64 *v41; // rsi
  __int64 v42; // rdx
  __int64 *v43; // rax
  int v44; // edx
  int v45; // ecx
  __int64 *v46; // r12
  int v47; // r15d
  int v48; // eax
  int v49; // edi
  int v50; // r11d
  __int64 *v51; // r10
  int v52; // ebx
  int v53; // edi
  int v54; // eax
  int v55; // eax
  int v56; // esi
  int v57; // edi
  __int64 v58; // [rsp+0h] [rbp-70h]
  int v60; // [rsp+14h] [rbp-5Ch]
  __int64 **v62; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v63; // [rsp+28h] [rbp-48h] BYREF
  __int64 **v64; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v65; // [rsp+38h] [rbp-38h]

  v3 = a3;
  v5 = *(__int64 **)a1;
  v62 = 0;
  if ( v5 != v3 )
  {
    if ( !v3 )
    {
      v46 = 0;
      goto LABEL_31;
    }
    while ( 1 )
    {
      v3 = (__int64 *)*v3;
      if ( v5 == v3 )
        break;
      if ( !v3 )
        goto LABEL_30;
    }
    v6 = (__int64)a3;
    v62 = (__int64 **)a3;
    for ( i = (__int64 *)*a3; v5 != i; i = (__int64 *)*i )
    {
      v62 = (__int64 **)i;
      v6 = (__int64)i;
    }
    v8 = *(_DWORD *)(a1 + 104);
    v64 = (__int64 **)v6;
    v65 = v5;
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 88);
      v10 = 1;
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v13 = (__int64 *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == v6 )
      {
LABEL_10:
        v46 = (__int64 *)v13[1];
        goto LABEL_11;
      }
      while ( v14 != -8 )
      {
        if ( !v11 && v14 == -16 )
          v11 = v13;
        v12 = (v8 - 1) & (v10 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( *v13 == v6 )
          goto LABEL_10;
        ++v10;
      }
      v54 = *(_DWORD *)(a1 + 96);
      if ( v11 )
        v13 = v11;
      ++*(_QWORD *)(a1 + 80);
      v55 = v54 + 1;
      if ( 4 * v55 < 3 * v8 )
      {
        if ( v8 - *(_DWORD *)(a1 + 100) - v55 > v8 >> 3 )
        {
LABEL_95:
          *(_DWORD *)(a1 + 96) = v55;
          if ( *v13 != -8 )
            --*(_DWORD *)(a1 + 100);
          *v13 = v6;
          v46 = v65;
          v13[1] = (__int64)v65;
LABEL_11:
          v15 = sub_157EBA0(a2);
          if ( !v15 )
            goto LABEL_33;
          goto LABEL_12;
        }
LABEL_100:
        sub_13FF990(a1 + 80, v8);
        sub_13FDE90(a1 + 80, (__int64 *)&v64, &v63);
        v13 = v63;
        v6 = (__int64)v64;
        v55 = *(_DWORD *)(a1 + 96) + 1;
        goto LABEL_95;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 80);
    }
    v8 *= 2;
    goto LABEL_100;
  }
LABEL_30:
  v46 = a3;
LABEL_31:
  v15 = sub_157EBA0(a2);
  if ( !v15 )
    return 0;
LABEL_12:
  v60 = sub_15F4D60(v15);
  if ( v60 )
  {
    v16 = 0;
    while ( 1 )
    {
      if ( a2 == sub_15F4DF0(v15, v16) )
        goto LABEL_24;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = sub_15F4DF0(v15, v16);
      v19 = *(_DWORD *)(v17 + 24);
      if ( !v19 )
      {
        v26 = *(__int64 **)a1;
        v63 = 0;
        if ( v26 )
          goto LABEL_35;
LABEL_56:
        *(_BYTE *)(a1 + 112) = 1;
        goto LABEL_24;
      }
      v20 = v19 - 1;
      v21 = *(_QWORD *)(v17 + 8);
      v22 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v18 == *v23 )
      {
LABEL_17:
        v25 = (__int64 *)v23[1];
      }
      else
      {
        v44 = 1;
        while ( v24 != -8 )
        {
          v45 = v44 + 1;
          v22 = v20 & (v44 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v18 == *v23 )
            goto LABEL_17;
          v44 = v45;
        }
        v25 = 0;
      }
      v26 = *(__int64 **)a1;
      v63 = v25;
      if ( v26 == v25 )
        goto LABEL_56;
      if ( !v25 )
        goto LABEL_35;
      v27 = v25;
      while ( 1 )
      {
        v27 = (__int64 *)*v27;
        if ( v26 == v27 )
          break;
        if ( !v27 )
          goto LABEL_42;
      }
      v28 = v62;
      if ( v62 )
        goto LABEL_24;
      v36 = *(_DWORD *)(a1 + 104);
      v58 = a1 + 80;
      if ( !v36 )
      {
        ++*(_QWORD *)(a1 + 80);
        goto LABEL_102;
      }
      v37 = *(_QWORD *)(a1 + 88);
      v38 = (v36 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v39 = (__int64 **)(v37 + 16LL * v38);
      v40 = *v39;
      if ( v25 != *v39 )
      {
        v47 = 1;
        while ( v40 != (__int64 *)-8LL )
        {
          if ( !v28 && v40 == (__int64 *)-16LL )
            v28 = v39;
          v57 = v47++;
          v38 = (v36 - 1) & (v57 + v38);
          v39 = (__int64 **)(v37 + 16LL * v38);
          v40 = *v39;
          if ( v25 == *v39 )
            goto LABEL_40;
        }
        v48 = *(_DWORD *)(a1 + 96);
        if ( !v28 )
          v28 = v39;
        ++*(_QWORD *)(a1 + 80);
        v49 = v48 + 1;
        if ( 4 * (v48 + 1) >= 3 * v36 )
        {
LABEL_102:
          v56 = 2 * v36;
        }
        else
        {
          if ( v36 - *(_DWORD *)(a1 + 100) - v49 > v36 >> 3 )
            goto LABEL_69;
          v56 = v36;
        }
        sub_13FF990(v58, v56);
        sub_13FDE90(v58, (__int64 *)&v63, &v64);
        v28 = v64;
        v25 = v63;
        v49 = *(_DWORD *)(a1 + 96) + 1;
LABEL_69:
        *(_DWORD *)(a1 + 96) = v49;
        if ( *v28 != (__int64 *)-8LL )
          --*(_DWORD *)(a1 + 100);
        *v28 = v25;
        v28[1] = 0;
        v26 = *(__int64 **)a1;
        v63 = 0;
        if ( !v26 )
          goto LABEL_24;
LABEL_35:
        if ( !v46 || v46 == v26 )
          v46 = 0;
        goto LABEL_24;
      }
LABEL_40:
      v25 = v39[1];
      v63 = v25;
      if ( v26 == v25 )
        goto LABEL_24;
      if ( !v25 )
        goto LABEL_35;
LABEL_42:
      if ( v26 != v25 )
      {
        if ( !v26 )
        {
LABEL_57:
          v42 = *v25;
          v63 = (__int64 *)v42;
          goto LABEL_48;
        }
        v41 = v26;
        while ( 1 )
        {
          v41 = (__int64 *)*v41;
          if ( v25 == v41 )
            break;
          if ( !v41 )
            goto LABEL_57;
        }
      }
      v42 = (__int64)v63;
LABEL_48:
      if ( v26 == v46 || !v46 )
        goto LABEL_55;
      if ( v46 != (__int64 *)v42 && v42 )
      {
        v43 = (__int64 *)v42;
        while ( 1 )
        {
          v43 = (__int64 *)*v43;
          if ( v43 == v46 )
            break;
          if ( !v43 )
            goto LABEL_24;
        }
LABEL_55:
        v46 = (__int64 *)v42;
      }
LABEL_24:
      if ( ++v16 == v60 )
        goto LABEL_25;
    }
  }
LABEL_33:
  v46 = 0;
LABEL_25:
  v29 = (__int64)v62;
  if ( v62 )
  {
    v30 = *(_DWORD *)(a1 + 104);
    if ( v30 )
    {
      v31 = *(_QWORD *)(a1 + 88);
      LODWORD(v32) = (v30 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v33 = (__int64 *)(v31 + 16LL * (unsigned int)v32);
      v34 = *v33;
      if ( v62 == (__int64 **)*v33 )
      {
LABEL_28:
        v33[1] = (__int64)v46;
        return a3;
      }
      v50 = 1;
      v51 = 0;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v51 )
          v51 = v33;
        v32 = (v30 - 1) & ((_DWORD)v32 + v50);
        v33 = (__int64 *)(v31 + 16 * v32);
        v34 = *v33;
        if ( v62 == (__int64 **)*v33 )
          goto LABEL_28;
        ++v50;
      }
      v52 = *(_DWORD *)(a1 + 96);
      if ( v51 )
        v33 = v51;
      ++*(_QWORD *)(a1 + 80);
      v53 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v30 )
      {
        if ( v30 - *(_DWORD *)(a1 + 100) - v53 > v30 >> 3 )
        {
LABEL_79:
          *(_DWORD *)(a1 + 96) = v53;
          if ( *v33 != -8 )
            --*(_DWORD *)(a1 + 100);
          *v33 = v29;
          v33[1] = 0;
          goto LABEL_28;
        }
LABEL_84:
        sub_13FF990(a1 + 80, v30);
        sub_13FDE90(a1 + 80, (__int64 *)&v62, &v64);
        v33 = (__int64 *)v64;
        v29 = (__int64)v62;
        v53 = *(_DWORD *)(a1 + 96) + 1;
        goto LABEL_79;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 80);
    }
    v30 *= 2;
    goto LABEL_84;
  }
  return v46;
}
