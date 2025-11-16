// Function: sub_25E2930
// Address: 0x25e2930
//
__int64 __fastcall sub_25E2930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r11
  __int64 i; // r15
  unsigned int v9; // r9d
  __int64 v10; // rcx
  unsigned int v11; // r8d
  __int64 *v12; // rax
  __int64 v13; // r11
  unsigned __int64 v14; // r8
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // r11
  unsigned __int64 v18; // rax
  __int64 v19; // r12
  __int64 *v20; // r14
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 *v24; // r10
  int v25; // edx
  __int64 *v26; // rcx
  __int64 v27; // r13
  __int64 v28; // r15
  unsigned int v29; // r9d
  __int64 v30; // rdi
  int v31; // r11d
  __int64 *v32; // r8
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // r10
  unsigned __int64 v36; // r8
  __int64 v37; // rcx
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r11
  unsigned __int64 v41; // rax
  __int64 *v42; // r14
  unsigned int v43; // esi
  __int64 v44; // rdx
  int v45; // ecx
  __int64 v46; // r12
  __int64 v47; // rax
  int v48; // eax
  __int64 *v49; // r10
  int v50; // eax
  int v52; // eax
  __int64 *v53; // r10
  int v54; // eax
  int v55; // eax
  int v56; // eax
  int v57; // [rsp+8h] [rbp-88h]
  unsigned __int64 v58; // [rsp+8h] [rbp-88h]
  int v59; // [rsp+8h] [rbp-88h]
  __int64 v61; // [rsp+18h] [rbp-78h]
  unsigned __int64 v62; // [rsp+20h] [rbp-70h]
  int v63; // [rsp+20h] [rbp-70h]
  __int64 v64; // [rsp+28h] [rbp-68h]
  __int64 v67; // [rsp+48h] [rbp-48h] BYREF
  __int64 v68; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v69[7]; // [rsp+58h] [rbp-38h] BYREF

  v5 = a1;
  v61 = a3 & 1;
  v64 = (a3 - 1) / 2;
  if ( a2 < v64 )
  {
    for ( i = a2; ; i = v19 )
    {
      v21 = *(_DWORD *)(a5 + 24);
      v19 = 2 * (i + 1);
      v20 = (__int64 *)(a1 + 16 * (i + 1));
      v22 = *(v20 - 1);
      v23 = *v20;
      v68 = v22;
      v67 = v23;
      if ( v21 )
      {
        v9 = v21 - 1;
        v10 = *(_QWORD *)(a5 + 8);
        v11 = (v21 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v23 == *v12 )
        {
LABEL_4:
          v14 = v12[1];
          goto LABEL_5;
        }
        v59 = 1;
        v24 = 0;
        while ( v13 != -4096 )
        {
          if ( v13 == -8192 && !v24 )
            v24 = v12;
          v11 = v9 & (v59 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v23 == *v12 )
            goto LABEL_4;
          ++v59;
        }
        if ( !v24 )
          v24 = v12;
        v56 = *(_DWORD *)(a5 + 16);
        ++*(_QWORD *)a5;
        v25 = v56 + 1;
        v69[0] = v24;
        if ( 4 * (v56 + 1) < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(a5 + 20) - v25 > v21 >> 3 )
            goto LABEL_73;
          goto LABEL_14;
        }
      }
      else
      {
        ++*(_QWORD *)a5;
        v69[0] = 0;
      }
      v21 *= 2;
LABEL_14:
      sub_9DDA50(a5, v21);
      sub_25E0C90(a5, &v67, v69);
      v23 = v67;
      v24 = (__int64 *)v69[0];
      v25 = *(_DWORD *)(a5 + 16) + 1;
LABEL_73:
      *(_DWORD *)(a5 + 16) = v25;
      if ( *v24 != -4096 )
        --*(_DWORD *)(a5 + 20);
      *v24 = v23;
      v24[1] = 0;
      v21 = *(_DWORD *)(a5 + 24);
      if ( !v21 )
      {
        ++*(_QWORD *)a5;
        v14 = 0;
        v69[0] = 0;
LABEL_77:
        v58 = v14;
        v21 *= 2;
LABEL_78:
        sub_9DDA50(a5, v21);
        sub_25E0C90(a5, &v68, v69);
        v22 = v68;
        v53 = (__int64 *)v69[0];
        v14 = v58;
        v55 = *(_DWORD *)(a5 + 16) + 1;
        goto LABEL_79;
      }
      v10 = *(_QWORD *)(a5 + 8);
      v22 = v68;
      v9 = v21 - 1;
      v14 = 0;
LABEL_5:
      v15 = v9 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v16 = (__int64 *)(v10 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v22 )
      {
LABEL_6:
        v18 = v16[1];
        goto LABEL_7;
      }
      v57 = 1;
      v53 = 0;
      while ( v17 != -4096 )
      {
        if ( !v53 && v17 == -8192 )
          v53 = v16;
        v15 = v9 & (v57 + v15);
        v16 = (__int64 *)(v10 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == v22 )
          goto LABEL_6;
        ++v57;
      }
      if ( !v53 )
        v53 = v16;
      v54 = *(_DWORD *)(a5 + 16);
      ++*(_QWORD *)a5;
      v55 = v54 + 1;
      v69[0] = v53;
      if ( 4 * v55 >= 3 * v21 )
        goto LABEL_77;
      if ( v21 - (v55 + *(_DWORD *)(a5 + 20)) <= v21 >> 3 )
      {
        v58 = v14;
        goto LABEL_78;
      }
LABEL_79:
      *(_DWORD *)(a5 + 16) = v55;
      if ( *v53 != -4096 )
        --*(_DWORD *)(a5 + 20);
      *v53 = v22;
      v18 = 0;
      v53[1] = 0;
LABEL_7:
      if ( v14 > v18 )
      {
        --v19;
        v20 = (__int64 *)(a1 + 8 * v19);
      }
      *(_QWORD *)(a1 + 8 * i) = *v20;
      if ( v19 >= v64 )
      {
        v26 = v20;
        v5 = a1;
        if ( v61 )
          goto LABEL_16;
        goto LABEL_30;
      }
    }
  }
  v19 = a2;
  v26 = (__int64 *)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
LABEL_30:
    if ( (a3 - 2) / 2 == v19 )
    {
      v46 = 2 * v19 + 2;
      v47 = *(_QWORD *)(v5 + 8 * v46 - 8);
      v19 = v46 - 1;
      *v26 = v47;
      v26 = (__int64 *)(v5 + 8 * v19);
    }
LABEL_16:
    v27 = (v19 - 1) / 2;
    if ( v19 > a2 )
    {
      v28 = v5;
      while ( 1 )
      {
        v42 = (__int64 *)(v28 + 8 * v27);
        v43 = *(_DWORD *)(a5 + 24);
        v44 = *v42;
        v68 = a4;
        v67 = v44;
        if ( v43 )
        {
          v29 = v43 - 1;
          v30 = *(_QWORD *)(a5 + 8);
          v31 = 1;
          v32 = 0;
          v33 = (v43 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v34 = (__int64 *)(v30 + 16LL * v33);
          v35 = *v34;
          if ( *v34 == v44 )
          {
LABEL_19:
            v36 = v34[1];
            v37 = a4;
            goto LABEL_20;
          }
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v32 )
              v32 = v34;
            v33 = v29 & (v31 + v33);
            v34 = (__int64 *)(v30 + 16LL * v33);
            v35 = *v34;
            if ( v44 == *v34 )
              goto LABEL_19;
            ++v31;
          }
          if ( !v32 )
            v32 = v34;
          v48 = *(_DWORD *)(a5 + 16);
          ++*(_QWORD *)a5;
          v45 = v48 + 1;
          v69[0] = v32;
          if ( 4 * (v48 + 1) < 3 * v43 )
          {
            if ( v43 - *(_DWORD *)(a5 + 20) - v45 > v43 >> 3 )
              goto LABEL_42;
            goto LABEL_28;
          }
        }
        else
        {
          ++*(_QWORD *)a5;
          v69[0] = 0;
        }
        v43 *= 2;
LABEL_28:
        sub_9DDA50(a5, v43);
        sub_25E0C90(a5, &v67, v69);
        v44 = v67;
        v32 = (__int64 *)v69[0];
        v45 = *(_DWORD *)(a5 + 16) + 1;
LABEL_42:
        *(_DWORD *)(a5 + 16) = v45;
        if ( *v32 != -4096 )
          --*(_DWORD *)(a5 + 20);
        *v32 = v44;
        v32[1] = 0;
        v43 = *(_DWORD *)(a5 + 24);
        if ( !v43 )
        {
          ++*(_QWORD *)a5;
          v36 = 0;
          v69[0] = 0;
LABEL_46:
          v62 = v36;
          v43 *= 2;
          goto LABEL_47;
        }
        v30 = *(_QWORD *)(a5 + 8);
        v37 = v68;
        v29 = v43 - 1;
        v36 = 0;
LABEL_20:
        v38 = v29 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v39 = (__int64 *)(v30 + 16LL * v38);
        v40 = *v39;
        if ( *v39 == v37 )
        {
LABEL_21:
          v41 = v39[1];
          goto LABEL_22;
        }
        v63 = 1;
        v49 = 0;
        while ( v40 != -4096 )
        {
          if ( !v49 && v40 == -8192 )
            v49 = v39;
          v38 = v29 & (v63 + v38);
          v39 = (__int64 *)(v30 + 16LL * v38);
          v40 = *v39;
          if ( *v39 == v37 )
            goto LABEL_21;
          ++v63;
        }
        if ( !v49 )
          v49 = v39;
        v52 = *(_DWORD *)(a5 + 16);
        ++*(_QWORD *)a5;
        v50 = v52 + 1;
        v69[0] = v49;
        if ( 4 * v50 >= 3 * v43 )
          goto LABEL_46;
        if ( v43 - (v50 + *(_DWORD *)(a5 + 20)) > v43 >> 3 )
          goto LABEL_57;
        v62 = v36;
LABEL_47:
        sub_9DDA50(a5, v43);
        sub_25E0C90(a5, &v68, v69);
        v37 = v68;
        v49 = (__int64 *)v69[0];
        v36 = v62;
        v50 = *(_DWORD *)(a5 + 16) + 1;
LABEL_57:
        *(_DWORD *)(a5 + 16) = v50;
        if ( *v49 != -4096 )
          --*(_DWORD *)(a5 + 20);
        *v49 = v37;
        v41 = 0;
        v49[1] = 0;
LABEL_22:
        v26 = (__int64 *)(v28 + 8 * v19);
        if ( v36 <= v41 )
          break;
        v19 = v27;
        *v26 = *v42;
        if ( a2 >= v27 )
        {
          v26 = (__int64 *)(v28 + 8 * v27);
          break;
        }
        v27 = (v27 - 1) / 2;
      }
    }
  }
  *v26 = a4;
  return a4;
}
