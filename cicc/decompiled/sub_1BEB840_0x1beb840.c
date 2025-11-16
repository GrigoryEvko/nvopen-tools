// Function: sub_1BEB840
// Address: 0x1beb840
//
__int64 __fastcall sub_1BEB840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  char v6; // al
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  _QWORD *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rdx
  int v21; // r8d
  __int64 v22; // r15
  int v23; // edi
  int v24; // edx
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r9
  unsigned int v28; // esi
  __int64 v29; // rdi
  int v30; // r11d
  __int64 v31; // r10
  int v32; // ecx
  int v33; // ecx
  __int64 v34; // rdi
  __int64 v35; // r9
  unsigned int v36; // r12d
  int v37; // r10d
  __int64 v38; // rsi
  __int64 *v39; // r12
  int v40; // eax
  __int64 *v41; // rbx
  int v42; // r8d
  int v43; // r9d
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 *v47; // r9
  __int64 v48; // rax
  int v49; // r8d
  __int64 *v50; // r9
  __int64 v51; // r12
  __int64 *v52; // r11
  __int64 v53; // r15
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // r14
  __int64 *v57; // r13
  __int64 *v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 *v64; // rdi
  const void *v65; // [rsp+8h] [rbp-A8h]
  __int64 v66; // [rsp+10h] [rbp-A0h]
  __int64 v67; // [rsp+18h] [rbp-98h]
  __int64 *v68; // [rsp+30h] [rbp-80h]
  __int64 v69; // [rsp+30h] [rbp-80h]
  char v70; // [rsp+3Fh] [rbp-71h]
  __int64 v71; // [rsp+40h] [rbp-70h]
  __int64 v72; // [rsp+48h] [rbp-68h]
  __int64 *v73; // [rsp+50h] [rbp-60h] BYREF
  __int64 v74; // [rsp+58h] [rbp-58h]
  _BYTE v75[80]; // [rsp+60h] [rbp-50h] BYREF

  result = a3 + 40;
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 40) = a2 + 112;
  v4 = *(_QWORD *)(a3 + 48);
  v71 = a3 + 40;
  if ( v4 != a3 + 40 )
  {
    v5 = a1;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        v72 = v4 - 24;
        v6 = *(_BYTE *)(v4 - 8);
        if ( v6 != 26 )
          break;
        result = *(_DWORD *)(v4 - 4) & 0xFFFFFFF;
        if ( (_DWORD)result == 3 )
          result = sub_1BEB4C0(v5, *(_QWORD *)(v4 - 96));
        v4 = *(_QWORD *)(v4 + 8);
        if ( v71 == v4 )
          return result;
      }
      if ( v6 == 77 )
      {
        v7 = sub_22077B0(120);
        v10 = (_QWORD *)v7;
        if ( v7 )
        {
          *(_BYTE *)(v7 + 40) = 2;
          *(_QWORD *)(v7 + 48) = v7 + 64;
          *(_QWORD *)(v7 + 56) = 0x100000000LL;
          *(_QWORD *)(v7 + 80) = v7 + 96;
          *(_QWORD *)(v7 + 88) = 0x200000000LL;
          *(_QWORD *)(v7 + 72) = 0;
          *(_QWORD *)(v7 + 8) = 0;
          *(_QWORD *)(v7 + 16) = 0;
          *(_BYTE *)(v7 + 24) = 2;
          *(_QWORD *)(v7 + 32) = 0;
          *(_QWORD *)v7 = &unk_49F7160;
          *(_BYTE *)(v7 + 112) = 53;
        }
        v11 = *(_QWORD *)(v5 + 32);
        if ( v11 )
        {
          v12 = *(unsigned __int64 **)(v5 + 40);
          v10[4] = v11;
          v13 = v10[1];
          v14 = *v12;
          v10[2] = v12;
          v14 &= 0xFFFFFFFFFFFFFFF8LL;
          v10[1] = v14 | v13 & 7;
          *(_QWORD *)(v14 + 8) = v10 + 1;
          *v12 = *v12 & 7 | (unsigned __int64)(v10 + 1);
        }
        v10[9] = v72;
        v15 = *(unsigned int *)(v5 + 120);
        if ( (unsigned int)v15 >= *(_DWORD *)(v5 + 124) )
        {
          sub_16CD150(v5 + 112, (const void *)(v5 + 128), 0, 8, v8, v9);
          v15 = *(unsigned int *)(v5 + 120);
        }
        v16 = (__int64)(v10 + 5);
        *(_QWORD *)(*(_QWORD *)(v5 + 112) + 8 * v15) = v72;
        ++*(_DWORD *)(v5 + 120);
      }
      else
      {
        v73 = (__int64 *)v75;
        v74 = 0x400000000LL;
        if ( (*(_BYTE *)(v4 - 1) & 0x40) != 0 )
        {
          v39 = *(__int64 **)(v4 - 32);
          v40 = *(_DWORD *)(v4 - 4);
        }
        else
        {
          v40 = *(_DWORD *)(v4 - 4);
          v39 = (__int64 *)(v72 - 24LL * (v40 & 0xFFFFFFF));
        }
        v41 = &v39[3 * (v40 & 0xFFFFFFF)];
        if ( v41 == v39 )
        {
          v47 = (__int64 *)v75;
          v46 = 0;
        }
        else
        {
          do
          {
            v44 = sub_1BEB4C0(v5, *v39);
            v45 = (unsigned int)v74;
            if ( (unsigned int)v74 >= HIDWORD(v74) )
            {
              sub_16CD150((__int64)&v73, v75, 0, 8, v42, v43);
              v45 = (unsigned int)v74;
            }
            v39 += 3;
            v73[v45] = v44;
            v46 = (unsigned int)(v74 + 1);
            LODWORD(v74) = v74 + 1;
          }
          while ( v41 != v39 );
          v47 = v73;
        }
        v68 = v47;
        v70 = *(_BYTE *)(v4 - 8);
        v48 = sub_22077B0(120);
        v50 = v68;
        v51 = v48;
        if ( v48 )
        {
          *(_BYTE *)(v48 + 40) = 2;
          v16 = v48 + 40;
          *(_QWORD *)(v48 + 48) = v48 + 64;
          v52 = &v68[v46];
          *(_QWORD *)(v48 + 56) = 0x100000000LL;
          *(_QWORD *)(v48 + 72) = 0;
          *(_QWORD *)(v48 + 80) = v48 + 96;
          *(_QWORD *)(v48 + 88) = 0x200000000LL;
          if ( v52 != v68 )
          {
            v69 = v4;
            v53 = *v50;
            v54 = v48 + 96;
            v67 = v5;
            v55 = 0;
            v56 = v48 + 40;
            v57 = v50;
            v66 = v48 + 80;
            v58 = v52;
            v65 = (const void *)(v48 + 96);
            while ( 1 )
            {
              *(_QWORD *)(v54 + 8 * v55) = v53;
              ++*(_DWORD *)(v51 + 88);
              v59 = *(unsigned int *)(v53 + 16);
              if ( (unsigned int)v59 >= *(_DWORD *)(v53 + 20) )
              {
                sub_16CD150(v53 + 8, (const void *)(v53 + 24), 0, 8, v49, (int)v50);
                v59 = *(unsigned int *)(v53 + 16);
              }
              ++v57;
              *(_QWORD *)(*(_QWORD *)(v53 + 8) + 8 * v59) = v56;
              ++*(_DWORD *)(v53 + 16);
              if ( v58 == v57 )
                break;
              v53 = *v57;
              v55 = *(unsigned int *)(v51 + 88);
              if ( (unsigned int)v55 >= *(_DWORD *)(v51 + 92) )
              {
                sub_16CD150(v66, v65, 0, 8, v49, (int)v50);
                v55 = *(unsigned int *)(v51 + 88);
              }
              v54 = *(_QWORD *)(v51 + 80);
            }
            v16 = v56;
            v4 = v69;
            v5 = v67;
          }
          *(_BYTE *)(v51 + 24) = 2;
          *(_QWORD *)(v51 + 8) = 0;
          *(_QWORD *)(v51 + 16) = 0;
          *(_QWORD *)v51 = &unk_49F7160;
          *(_QWORD *)(v51 + 32) = 0;
          *(_BYTE *)(v51 + 112) = v70 - 24;
        }
        else
        {
          v16 = 40;
        }
        v60 = *(_QWORD *)(v5 + 32);
        if ( v60 )
        {
          v61 = *(__int64 **)(v5 + 40);
          *(_QWORD *)(v51 + 32) = v60;
          v62 = *(_QWORD *)(v51 + 8);
          v63 = *v61;
          *(_QWORD *)(v51 + 16) = v61;
          v63 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v51 + 8) = v63 | v62 & 7;
          *(_QWORD *)(v63 + 8) = v51 + 8;
          *v61 = *v61 & 7 | (v51 + 8);
        }
        v64 = v73;
        *(_QWORD *)(v51 + 72) = v72;
        if ( v64 != (__int64 *)v75 )
          _libc_free((unsigned __int64)v64);
      }
      v17 = *(_DWORD *)(v5 + 104);
      if ( !v17 )
        break;
      v18 = *(_QWORD *)(v5 + 88);
      v19 = (v17 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      result = v18 + 16LL * v19;
      v20 = *(_QWORD *)result;
      if ( v72 != *(_QWORD *)result )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = result;
          v19 = (v17 - 1) & (v21 + v19);
          result = v18 + 16LL * v19;
          v20 = *(_QWORD *)result;
          if ( v72 == *(_QWORD *)result )
            goto LABEL_18;
          ++v21;
        }
        v23 = *(_DWORD *)(v5 + 96);
        if ( v22 )
          result = v22;
        ++*(_QWORD *)(v5 + 80);
        v24 = v23 + 1;
        if ( 4 * (v23 + 1) < 3 * v17 )
        {
          if ( v17 - *(_DWORD *)(v5 + 100) - v24 <= v17 >> 3 )
          {
            sub_1BA21E0(v5 + 80, v17);
            v32 = *(_DWORD *)(v5 + 104);
            if ( !v32 )
            {
LABEL_83:
              ++*(_DWORD *)(v5 + 96);
              BUG();
            }
            v33 = v32 - 1;
            v34 = *(_QWORD *)(v5 + 88);
            v35 = 0;
            v36 = v33 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
            v37 = 1;
            v24 = *(_DWORD *)(v5 + 96) + 1;
            result = v34 + 16LL * v36;
            v38 = *(_QWORD *)result;
            if ( *(_QWORD *)result != v72 )
            {
              while ( v38 != -8 )
              {
                if ( v38 == -16 && !v35 )
                  v35 = result;
                v36 = v33 & (v37 + v36);
                result = v34 + 16LL * v36;
                v38 = *(_QWORD *)result;
                if ( v72 == *(_QWORD *)result )
                  goto LABEL_26;
                ++v37;
              }
              if ( v35 )
                result = v35;
            }
          }
          goto LABEL_26;
        }
LABEL_30:
        sub_1BA21E0(v5 + 80, 2 * v17);
        v25 = *(_DWORD *)(v5 + 104);
        if ( !v25 )
          goto LABEL_83;
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v5 + 88);
        v24 = *(_DWORD *)(v5 + 96) + 1;
        v28 = v26 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        result = v27 + 16LL * v28;
        v29 = *(_QWORD *)result;
        if ( v72 != *(_QWORD *)result )
        {
          v30 = 1;
          v31 = 0;
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v31 )
              v31 = result;
            v28 = v26 & (v30 + v28);
            result = v27 + 16LL * v28;
            v29 = *(_QWORD *)result;
            if ( v72 == *(_QWORD *)result )
              goto LABEL_26;
            ++v30;
          }
          if ( v31 )
            result = v31;
        }
LABEL_26:
        *(_DWORD *)(v5 + 96) = v24;
        if ( *(_QWORD *)result != -8 )
          --*(_DWORD *)(v5 + 100);
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)result = v72;
      }
LABEL_18:
      *(_QWORD *)(result + 8) = v16;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v71 == v4 )
        return result;
    }
    ++*(_QWORD *)(v5 + 80);
    goto LABEL_30;
  }
  return result;
}
