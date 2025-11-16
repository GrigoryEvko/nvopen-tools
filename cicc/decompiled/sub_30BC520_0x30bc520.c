// Function: sub_30BC520
// Address: 0x30bc520
//
__int64 __fastcall sub_30BC520(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r8
  int v6; // r11d
  __int64 *v7; // rdx
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // esi
  __int64 v12; // r14
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r12
  __int64 (__fastcall *v18)(__int64, __int64); // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // rcx
  unsigned int v26; // esi
  int v27; // eax
  int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // eax
  int v31; // ecx
  __int64 v32; // r8
  int v33; // eax
  int v34; // edi
  __int64 v35; // rsi
  unsigned int v36; // eax
  int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r8
  int v40; // r10d
  __int64 *v41; // r9
  int v42; // eax
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  __int64 *v46; // r8
  unsigned int v47; // r14d
  int v48; // r9d
  __int64 v49; // rsi
  __int64 v50; // rax
  int v51; // eax
  int v52; // edi
  __int64 v53; // rsi
  unsigned int v54; // eax
  int v55; // ecx
  __int64 *v56; // rdx
  __int64 v57; // r8
  int v58; // r10d
  __int64 *v59; // r9
  int v60; // r10d
  int v61; // eax
  int v62; // r10d
  int v63; // eax
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdi
  __int64 *v67; // r8
  __int64 v68; // r14
  int v69; // r9d
  __int64 v70; // rsi
  int v71; // eax
  int v72; // eax
  __int64 v73; // rdi
  __int64 *v74; // r8
  unsigned int v75; // r12d
  int v76; // r9d
  __int64 v77; // rsi
  int v78; // r10d
  __int64 *v79; // r9
  __int64 v80; // [rsp+8h] [rbp-58h]
  __int64 v81; // [rsp+10h] [rbp-50h]
  __int64 v82; // [rsp+18h] [rbp-48h]
  __int64 v83; // [rsp+20h] [rbp-40h]
  __int64 v84; // [rsp+28h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 24);
  v81 = a1 + 32;
  v80 = a1 + 96;
  result = *v1;
  v83 = *v1;
  v82 = *v1 + 8LL * *((unsigned int *)v1 + 2);
  if ( *v1 != v82 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)v83 + 56LL);
      v84 = *(_QWORD *)v83 + 48LL;
      if ( v84 != v4 )
        break;
LABEL_55:
      v83 += 8;
      result = v83;
      if ( v82 == v83 )
        return result;
    }
    while ( 1 )
    {
      v17 = v4 - 24;
      if ( !v4 )
        v17 = 0;
      v18 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 24LL);
      if ( v18 == sub_30B2820 )
      {
        v19 = sub_22077B0(0x60u);
        v20 = v19;
        if ( v19 )
          sub_30B0D20(v19, v17);
        sub_30B2450(*(_QWORD *)(a1 + 8), v20);
        v21 = *(_DWORD *)(a1 + 56);
        if ( !v21 )
        {
LABEL_47:
          ++*(_QWORD *)(a1 + 32);
          goto LABEL_48;
        }
      }
      else
      {
        v50 = v18(a1, v17);
        v21 = *(_DWORD *)(a1 + 56);
        v20 = v50;
        if ( !v21 )
          goto LABEL_47;
      }
      v22 = *(_QWORD *)(a1 + 40);
      v23 = (v21 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v24 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( v17 == *v24 )
        goto LABEL_15;
      v62 = 1;
      v56 = 0;
      while ( v25 != -4096 )
      {
        if ( v25 != -8192 || v56 )
          v24 = v56;
        v23 = (v21 - 1) & (v62 + v23);
        v25 = *(_QWORD *)(v22 + 16LL * v23);
        if ( v17 == v25 )
          goto LABEL_15;
        ++v62;
        v56 = v24;
        v24 = (__int64 *)(v22 + 16LL * v23);
      }
      if ( !v56 )
        v56 = v24;
      v63 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v55 = v63 + 1;
      if ( 4 * (v63 + 1) < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(a1 + 52) - v55 <= v21 >> 3 )
        {
          sub_30BC340(v81, v21);
          v64 = *(_DWORD *)(a1 + 56);
          if ( !v64 )
          {
LABEL_133:
            ++*(_DWORD *)(a1 + 48);
            BUG();
          }
          v65 = v64 - 1;
          v66 = *(_QWORD *)(a1 + 40);
          v67 = 0;
          LODWORD(v68) = v65 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v69 = 1;
          v55 = *(_DWORD *)(a1 + 48) + 1;
          v56 = (__int64 *)(v66 + 16LL * (unsigned int)v68);
          v70 = *v56;
          if ( v17 != *v56 )
          {
            while ( v70 != -4096 )
            {
              if ( !v67 && v70 == -8192 )
                v67 = v56;
              v68 = v65 & (unsigned int)(v68 + v69);
              v56 = (__int64 *)(v66 + 16 * v68);
              v70 = *v56;
              if ( v17 == *v56 )
                goto LABEL_72;
              ++v69;
            }
            if ( v67 )
              v56 = v67;
          }
        }
        goto LABEL_72;
      }
LABEL_48:
      sub_30BC340(v81, 2 * v21);
      v51 = *(_DWORD *)(a1 + 56);
      if ( !v51 )
        goto LABEL_133;
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 40);
      v54 = (v51 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v55 = *(_DWORD *)(a1 + 48) + 1;
      v56 = (__int64 *)(v53 + 16LL * v54);
      v57 = *v56;
      if ( v17 != *v56 )
      {
        v58 = 1;
        v59 = 0;
        while ( v57 != -4096 )
        {
          if ( v57 == -8192 && !v59 )
            v59 = v56;
          v54 = v52 & (v58 + v54);
          v56 = (__int64 *)(v53 + 16LL * v54);
          v57 = *v56;
          if ( v17 == *v56 )
            goto LABEL_72;
          ++v58;
        }
        if ( v59 )
          v56 = v59;
      }
LABEL_72:
      *(_DWORD *)(a1 + 48) = v55;
      if ( *v56 != -4096 )
        --*(_DWORD *)(a1 + 52);
      *v56 = v17;
      v56[1] = v20;
LABEL_15:
      v26 = *(_DWORD *)(a1 + 88);
      if ( !v26 )
      {
        ++*(_QWORD *)(a1 + 64);
LABEL_17:
        sub_30BB7B0(a1 + 64, 2 * v26);
        v27 = *(_DWORD *)(a1 + 88);
        if ( !v27 )
          goto LABEL_132;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 72);
        v30 = (v27 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v31 = *(_DWORD *)(a1 + 80) + 1;
        v7 = (__int64 *)(v29 + 16LL * v30);
        v32 = *v7;
        if ( v17 != *v7 )
        {
          v78 = 1;
          v79 = 0;
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v79 )
              v79 = v7;
            v30 = v28 & (v30 + v78);
            v7 = (__int64 *)(v29 + 16LL * v30);
            v32 = *v7;
            if ( v17 == *v7 )
              goto LABEL_19;
            ++v78;
          }
          if ( v79 )
            v7 = v79;
        }
        goto LABEL_19;
      }
      v5 = *(_QWORD *)(a1 + 72);
      v6 = 1;
      v7 = 0;
      v8 = (v26 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v17 == *v9 )
      {
LABEL_5:
        v11 = *(_DWORD *)(a1 + 120);
        v12 = v9[1];
        if ( !v11 )
          goto LABEL_22;
        goto LABEL_6;
      }
      while ( v10 != -4096 )
      {
        if ( v10 == -8192 && !v7 )
          v7 = v9;
        v8 = (v26 - 1) & (v6 + v8);
        v9 = (__int64 *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( v17 == *v9 )
          goto LABEL_5;
        ++v6;
      }
      if ( !v7 )
        v7 = v9;
      v42 = *(_DWORD *)(a1 + 80);
      ++*(_QWORD *)(a1 + 64);
      v31 = v42 + 1;
      if ( 4 * (v42 + 1) >= 3 * v26 )
        goto LABEL_17;
      if ( v26 - *(_DWORD *)(a1 + 84) - v31 <= v26 >> 3 )
      {
        sub_30BB7B0(a1 + 64, v26);
        v43 = *(_DWORD *)(a1 + 88);
        if ( !v43 )
        {
LABEL_132:
          ++*(_DWORD *)(a1 + 80);
          BUG();
        }
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 72);
        v46 = 0;
        v47 = v44 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v48 = 1;
        v31 = *(_DWORD *)(a1 + 80) + 1;
        v7 = (__int64 *)(v45 + 16LL * v47);
        v49 = *v7;
        if ( v17 != *v7 )
        {
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v46 )
              v46 = v7;
            v47 = v44 & (v48 + v47);
            v7 = (__int64 *)(v45 + 16LL * v47);
            v49 = *v7;
            if ( v17 == *v7 )
              goto LABEL_19;
            ++v48;
          }
          if ( v46 )
            v7 = v46;
        }
      }
LABEL_19:
      *(_DWORD *)(a1 + 80) = v31;
      if ( *v7 != -4096 )
        --*(_DWORD *)(a1 + 84);
      *v7 = v17;
      v12 = 0;
      v7[1] = 0;
      v11 = *(_DWORD *)(a1 + 120);
      if ( !v11 )
      {
LABEL_22:
        ++*(_QWORD *)(a1 + 96);
        goto LABEL_23;
      }
LABEL_6:
      v13 = *(_QWORD *)(a1 + 104);
      v14 = (v11 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v20 != *v15 )
      {
        v60 = 1;
        v38 = 0;
        while ( v16 != -4096 )
        {
          if ( v38 || v16 != -8192 )
            v15 = v38;
          v14 = (v11 - 1) & (v60 + v14);
          v16 = *(_QWORD *)(v13 + 16LL * v14);
          if ( v20 == v16 )
            goto LABEL_7;
          ++v60;
          v38 = v15;
          v15 = (__int64 *)(v13 + 16LL * v14);
        }
        if ( !v38 )
          v38 = v15;
        v61 = *(_DWORD *)(a1 + 112);
        ++*(_QWORD *)(a1 + 96);
        v37 = v61 + 1;
        if ( 4 * (v61 + 1) >= 3 * v11 )
        {
LABEL_23:
          sub_30BBCC0(v80, 2 * v11);
          v33 = *(_DWORD *)(a1 + 120);
          if ( !v33 )
            goto LABEL_134;
          v34 = v33 - 1;
          v35 = *(_QWORD *)(a1 + 104);
          v36 = (v33 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v37 = *(_DWORD *)(a1 + 112) + 1;
          v38 = (__int64 *)(v35 + 16LL * v36);
          v39 = *v38;
          if ( v20 != *v38 )
          {
            v40 = 1;
            v41 = 0;
            while ( v39 != -4096 )
            {
              if ( !v41 && v39 == -8192 )
                v41 = v38;
              v36 = v34 & (v40 + v36);
              v38 = (__int64 *)(v35 + 16LL * v36);
              v39 = *v38;
              if ( v20 == *v38 )
                goto LABEL_63;
              ++v40;
            }
            if ( v41 )
              v38 = v41;
          }
        }
        else if ( v11 - *(_DWORD *)(a1 + 116) - v37 <= v11 >> 3 )
        {
          sub_30BBCC0(v80, v11);
          v71 = *(_DWORD *)(a1 + 120);
          if ( !v71 )
          {
LABEL_134:
            ++*(_DWORD *)(a1 + 112);
            BUG();
          }
          v72 = v71 - 1;
          v73 = *(_QWORD *)(a1 + 104);
          v74 = 0;
          v75 = v72 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v76 = 1;
          v37 = *(_DWORD *)(a1 + 112) + 1;
          v38 = (__int64 *)(v73 + 16LL * v75);
          v77 = *v38;
          if ( v20 != *v38 )
          {
            while ( v77 != -4096 )
            {
              if ( v77 == -8192 && !v74 )
                v74 = v38;
              v75 = v72 & (v75 + v76);
              v38 = (__int64 *)(v73 + 16LL * v75);
              v77 = *v38;
              if ( v20 == *v38 )
                goto LABEL_63;
              ++v76;
            }
            if ( v74 )
              v38 = v74;
          }
        }
LABEL_63:
        *(_DWORD *)(a1 + 112) = v37;
        if ( *v38 != -4096 )
          --*(_DWORD *)(a1 + 116);
        *v38 = v20;
        v38[1] = v12;
      }
LABEL_7:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v84 == v4 )
        goto LABEL_55;
    }
  }
  return result;
}
