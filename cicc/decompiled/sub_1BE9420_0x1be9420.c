// Function: sub_1BE9420
// Address: 0x1be9420
//
__int64 __fastcall sub_1BE9420(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 v9; // rcx
  unsigned int v10; // esi
  __int64 v11; // r12
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // r8
  __int64 v15; // rsi
  int v16; // eax
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned int v22; // edx
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r14
  _BYTE *v26; // rsi
  unsigned int v27; // esi
  __int64 v28; // r9
  unsigned int v29; // r8d
  __int64 v30; // rdi
  __int64 v31; // r12
  __int64 v32; // rdi
  __int64 v33; // r12
  __int64 v34; // rdi
  int v35; // r10d
  int v36; // eax
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // rdi
  int v40; // r9d
  unsigned int v41; // r14d
  _QWORD *v42; // r8
  __int64 v43; // rsi
  int v44; // esi
  int v45; // esi
  __int64 v46; // r8
  unsigned int v47; // edx
  _QWORD *v48; // rcx
  __int64 v49; // rdi
  int v50; // r11d
  _QWORD *v51; // r9
  int v52; // ecx
  int v53; // r11d
  int v54; // eax
  int v55; // esi
  int v56; // esi
  __int64 v57; // r8
  int v58; // r11d
  unsigned int v59; // edx
  __int64 v60; // rdi
  int v61; // r9d
  int v62; // r10d
  _QWORD *v63; // r9
  __int64 v64; // [rsp+8h] [rbp-58h]
  __int64 v65; // [rsp+10h] [rbp-50h]
  unsigned int v66; // [rsp+10h] [rbp-50h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+20h] [rbp-40h] BYREF
  __int64 v69[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_1BE8E40(a1 + 24, (__int64 *)(*(_QWORD *)a1 + 8LL))[4] = *a3;
  result = *(_QWORD *)a1;
  v67 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  if ( v67 != 1 )
  {
    v6 = 1;
    v64 = a2 + 24;
    while ( 1 )
    {
      v10 = *(_DWORD *)(a2 + 48);
      v11 = *(_QWORD *)(result + 8 * v6);
      if ( !v10 )
        break;
      v7 = *(_QWORD *)(a2 + 32);
      v8 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      result = v7 + 16LL * v8;
      v9 = *(_QWORD *)result;
      if ( v11 != *(_QWORD *)result )
      {
        v35 = 1;
        v17 = 0;
        while ( v9 != -8 )
        {
          if ( !v17 && v9 == -16 )
            v17 = (_QWORD *)result;
          v8 = (v10 - 1) & (v35 + v8);
          result = v7 + 16LL * v8;
          v9 = *(_QWORD *)result;
          if ( v11 == *(_QWORD *)result )
            goto LABEL_4;
          ++v35;
        }
        if ( !v17 )
          v17 = (_QWORD *)result;
        v36 = *(_DWORD *)(a2 + 40);
        ++*(_QWORD *)(a2 + 24);
        v16 = v36 + 1;
        if ( 4 * v16 < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a2 + 44) - v16 <= v10 >> 3 )
          {
            sub_1BE8590(v64, v10);
            v37 = *(_DWORD *)(a2 + 48);
            if ( !v37 )
              goto LABEL_105;
            v38 = v37 - 1;
            v39 = *(_QWORD *)(a2 + 32);
            v40 = 1;
            v41 = v38 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v42 = 0;
            v16 = *(_DWORD *)(a2 + 40) + 1;
            v17 = (_QWORD *)(v39 + 16LL * v41);
            v43 = *v17;
            if ( v11 != *v17 )
            {
              while ( v43 != -8 )
              {
                if ( v43 == -16 && !v42 )
                  v42 = v17;
                v41 = v38 & (v40 + v41);
                v17 = (_QWORD *)(v39 + 16LL * v41);
                v43 = *v17;
                if ( v11 == *v17 )
                  goto LABEL_11;
                ++v40;
              }
              if ( v42 )
                v17 = v42;
            }
          }
          goto LABEL_11;
        }
LABEL_9:
        sub_1BE8590(v64, 2 * v10);
        v12 = *(_DWORD *)(a2 + 48);
        if ( !v12 )
          goto LABEL_105;
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a2 + 32);
        LODWORD(v15) = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = *(_DWORD *)(a2 + 40) + 1;
        v17 = (_QWORD *)(v14 + 16LL * (unsigned int)v15);
        v18 = *v17;
        if ( v11 != *v17 )
        {
          v62 = 1;
          v63 = 0;
          while ( v18 != -8 )
          {
            if ( !v63 && v18 == -16 )
              v63 = v17;
            v15 = v13 & (unsigned int)(v15 + v62);
            v17 = (_QWORD *)(v14 + 16 * v15);
            v18 = *v17;
            if ( v11 == *v17 )
              goto LABEL_11;
            ++v62;
          }
          if ( v63 )
            v17 = v63;
        }
LABEL_11:
        *(_DWORD *)(a2 + 40) = v16;
        if ( *v17 != -8 )
          --*(_DWORD *)(a2 + 44);
        *v17 = v11;
        v17[1] = 0;
        goto LABEL_14;
      }
LABEL_4:
      if ( *(_QWORD *)(result + 8) )
        goto LABEL_5;
LABEL_14:
      v19 = *(unsigned int *)(a1 + 48);
      v20 = 0;
      if ( !(_DWORD)v19 )
        goto LABEL_18;
      v21 = *(_QWORD *)(a1 + 32);
      v22 = (v19 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v23 = (__int64 *)(v21 + 72LL * v22);
      v24 = *v23;
      if ( v11 == *v23 )
      {
LABEL_16:
        if ( v23 != (__int64 *)(v21 + 72 * v19) )
        {
          v20 = v23[4];
          goto LABEL_18;
        }
      }
      else
      {
        v52 = 1;
        while ( v24 != -8 )
        {
          v61 = v52 + 1;
          v22 = (v19 - 1) & (v52 + v22);
          v23 = (__int64 *)(v21 + 72LL * v22);
          v24 = *v23;
          if ( v11 == *v23 )
            goto LABEL_16;
          v52 = v61;
        }
      }
      v20 = 0;
LABEL_18:
      v65 = sub_1BE87B0(a1, v20, a2);
      sub_1BE2190(&v68, v11, v65);
      v25 = v68;
      v69[0] = v68;
      v26 = *(_BYTE **)(v65 + 32);
      if ( v26 == *(_BYTE **)(v65 + 40) )
      {
        sub_1BE72B0(v65 + 24, v26, v69);
        v27 = *(_DWORD *)(a2 + 48);
        v25 = v68;
        v68 = 0;
        if ( !v27 )
          goto LABEL_45;
      }
      else
      {
        if ( v26 )
        {
          *(_QWORD *)v26 = v68;
          v26 = *(_BYTE **)(v65 + 32);
          v25 = v68;
        }
        v68 = 0;
        *(_QWORD *)(v65 + 32) = v26 + 8;
        v27 = *(_DWORD *)(a2 + 48);
        if ( !v27 )
        {
LABEL_45:
          ++*(_QWORD *)(a2 + 24);
          goto LABEL_46;
        }
      }
      v28 = *(_QWORD *)(a2 + 32);
      v29 = (v27 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      result = v28 + 16LL * v29;
      v30 = *(_QWORD *)result;
      if ( v11 == *(_QWORD *)result )
      {
LABEL_23:
        v31 = *(_QWORD *)(result + 8);
        *(_QWORD *)(result + 8) = v25;
        if ( !v31 )
          goto LABEL_5;
        v32 = *(_QWORD *)(v31 + 24);
        if ( v32 )
          j_j___libc_free_0(v32, *(_QWORD *)(v31 + 40) - v32);
        result = j_j___libc_free_0(v31, 56);
        goto LABEL_27;
      }
      v53 = 1;
      v48 = 0;
      while ( v30 != -8 )
      {
        if ( !v48 && v30 == -16 )
          v48 = (_QWORD *)result;
        v29 = (v27 - 1) & (v53 + v29);
        result = v28 + 16LL * v29;
        v30 = *(_QWORD *)result;
        if ( v11 == *(_QWORD *)result )
          goto LABEL_23;
        ++v53;
      }
      if ( !v48 )
        v48 = (_QWORD *)result;
      v54 = *(_DWORD *)(a2 + 40);
      ++*(_QWORD *)(a2 + 24);
      result = (unsigned int)(v54 + 1);
      if ( 4 * (int)result < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(a2 + 44) - (unsigned int)result > v27 >> 3 )
          goto LABEL_67;
        v66 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
        sub_1BE8590(v64, v27);
        v55 = *(_DWORD *)(a2 + 48);
        if ( !v55 )
        {
LABEL_105:
          ++*(_DWORD *)(a2 + 40);
          BUG();
        }
        v56 = v55 - 1;
        v57 = *(_QWORD *)(a2 + 32);
        v58 = 1;
        v51 = 0;
        v59 = v56 & v66;
        result = (unsigned int)(*(_DWORD *)(a2 + 40) + 1);
        v48 = (_QWORD *)(v57 + 16LL * (v56 & v66));
        v60 = *v48;
        if ( v11 == *v48 )
          goto LABEL_67;
        while ( v60 != -8 )
        {
          if ( !v51 && v60 == -16 )
            v51 = v48;
          v59 = v56 & (v58 + v59);
          v48 = (_QWORD *)(v57 + 16LL * v59);
          v60 = *v48;
          if ( v11 == *v48 )
            goto LABEL_67;
          ++v58;
        }
        goto LABEL_50;
      }
LABEL_46:
      sub_1BE8590(v64, 2 * v27);
      v44 = *(_DWORD *)(a2 + 48);
      if ( !v44 )
        goto LABEL_105;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a2 + 32);
      v47 = v45 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      result = (unsigned int)(*(_DWORD *)(a2 + 40) + 1);
      v48 = (_QWORD *)(v46 + 16LL * v47);
      v49 = *v48;
      if ( v11 == *v48 )
        goto LABEL_67;
      v50 = 1;
      v51 = 0;
      while ( v49 != -8 )
      {
        if ( v49 == -16 && !v51 )
          v51 = v48;
        v47 = v45 & (v50 + v47);
        v48 = (_QWORD *)(v46 + 16LL * v47);
        v49 = *v48;
        if ( v11 == *v48 )
          goto LABEL_67;
        ++v50;
      }
LABEL_50:
      if ( v51 )
        v48 = v51;
LABEL_67:
      *(_DWORD *)(a2 + 40) = result;
      if ( *v48 != -8 )
        --*(_DWORD *)(a2 + 44);
      *v48 = v11;
      v48[1] = v25;
LABEL_27:
      v33 = v68;
      if ( !v68 )
      {
LABEL_5:
        if ( v67 == ++v6 )
          return result;
        goto LABEL_6;
      }
      v34 = *(_QWORD *)(v68 + 24);
      if ( v34 )
        j_j___libc_free_0(v34, *(_QWORD *)(v68 + 40) - v34);
      ++v6;
      result = j_j___libc_free_0(v33, 56);
      if ( v67 == v6 )
        return result;
LABEL_6:
      result = *(_QWORD *)a1;
    }
    ++*(_QWORD *)(a2 + 24);
    goto LABEL_9;
  }
  return result;
}
