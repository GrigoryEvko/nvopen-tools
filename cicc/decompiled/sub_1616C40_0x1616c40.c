// Function: sub_1616C40
// Address: 0x1616c40
//
__int64 __fastcall sub_1616C40(__int64 a1)
{
  __int64 *v2; // r13
  __int64 *i; // r15
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *j; // rdx
  __int64 *v9; // r13
  __int64 result; // rax
  __int64 *m; // r15
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // rdx
  __int64 n; // rdx
  unsigned int v16; // ecx
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // r8
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *k; // rdx
  unsigned int v26; // ecx
  _QWORD *v27; // rdi
  unsigned int v28; // eax
  int v29; // eax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 ii; // rdx
  __int64 *v35; // r13
  __int64 v36; // r12
  __int64 *v37; // rbx
  unsigned int v38; // esi
  __int64 v39; // r15
  __int64 v40; // r8
  unsigned int v41; // edi
  __int64 v42; // rcx
  _QWORD *v43; // rdx
  __int64 v44; // r8
  int v45; // edx
  int v46; // edx
  __int64 v47; // r10
  unsigned int v48; // esi
  int v49; // ecx
  __int64 v50; // rdi
  int v51; // r9d
  __int64 v52; // r8
  _QWORD *v53; // rdi
  unsigned int v54; // r9d
  _QWORD *v55; // rsi
  int v56; // r11d
  __int64 v57; // r10
  int v58; // edi
  int v59; // esi
  int v60; // esi
  __int64 v61; // r10
  int v62; // r9d
  unsigned int v63; // edx
  __int64 v64; // rdi
  _QWORD *v65; // rax
  int v66; // [rsp+4h] [rbp-3Ch]
  int v67; // [rsp+4h] [rbp-3Ch]
  unsigned int v68; // [rsp+4h] [rbp-3Ch]
  __int64 v69; // [rsp+8h] [rbp-38h]
  __int64 v70; // [rsp+8h] [rbp-38h]
  __int64 v71; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 32);
  for ( i = &v2[*(unsigned int *)(a1 + 40)]; i != v2; *(_QWORD *)(v4 + 216) = 0 )
  {
    v4 = *v2;
    v5 = *(_DWORD *)(*v2 + 240);
    ++*(_QWORD *)(*v2 + 224);
    if ( v5 )
    {
      v16 = 4 * v5;
      v6 = *(unsigned int *)(v4 + 248);
      if ( (unsigned int)(4 * v5) < 0x40 )
        v16 = 64;
      if ( v16 >= (unsigned int)v6 )
      {
LABEL_5:
        v7 = *(_QWORD **)(v4 + 232);
        for ( j = &v7[2 * v6]; j != v7; v7 += 2 )
          *v7 = -4;
        *(_QWORD *)(v4 + 240) = 0;
        goto LABEL_8;
      }
      v17 = *(_QWORD **)(v4 + 232);
      v18 = v5 - 1;
      if ( v18 )
      {
        _BitScanReverse(&v18, v18);
        v19 = 1 << (33 - (v18 ^ 0x1F));
        if ( v19 < 64 )
          v19 = 64;
        if ( (_DWORD)v6 == v19 )
        {
          *(_QWORD *)(v4 + 240) = 0;
          v65 = &v17[2 * (unsigned int)v6];
          do
          {
            if ( v17 )
              *v17 = -4;
            v17 += 2;
          }
          while ( v65 != v17 );
          goto LABEL_8;
        }
        v20 = (4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1);
        v21 = ((v20 | (v20 >> 2)) >> 4) | v20 | (v20 >> 2) | ((((v20 | (v20 >> 2)) >> 4) | v20 | (v20 >> 2)) >> 8);
        v66 = (v21 | (v21 >> 16)) + 1;
        v22 = 16 * ((v21 | (v21 >> 16)) + 1);
      }
      else
      {
        v66 = 128;
        v22 = 2048;
      }
      v69 = v22;
      j___libc_free_0(v17);
      *(_DWORD *)(v4 + 248) = v66;
      v23 = (_QWORD *)sub_22077B0(v69);
      v24 = *(unsigned int *)(v4 + 248);
      *(_QWORD *)(v4 + 240) = 0;
      *(_QWORD *)(v4 + 232) = v23;
      for ( k = &v23[2 * v24]; k != v23; v23 += 2 )
      {
        if ( v23 )
          *v23 = -4;
      }
    }
    else if ( *(_DWORD *)(v4 + 244) )
    {
      v6 = *(unsigned int *)(v4 + 248);
      if ( (unsigned int)v6 <= 0x40 )
        goto LABEL_5;
      j___libc_free_0(*(_QWORD *)(v4 + 232));
      *(_QWORD *)(v4 + 232) = 0;
      *(_QWORD *)(v4 + 240) = 0;
      *(_DWORD *)(v4 + 248) = 0;
    }
LABEL_8:
    *(_QWORD *)(v4 + 168) = 0;
    ++v2;
    *(_QWORD *)(v4 + 176) = 0;
    *(_QWORD *)(v4 + 184) = 0;
    *(_QWORD *)(v4 + 192) = 0;
    *(_QWORD *)(v4 + 200) = 0;
    *(_QWORD *)(v4 + 208) = 0;
  }
  v9 = *(__int64 **)(a1 + 112);
  result = *(unsigned int *)(a1 + 120);
  for ( m = &v9[result]; m != v9; *(_QWORD *)(v12 + 216) = 0 )
  {
    v12 = *v9;
    v13 = *(_DWORD *)(*v9 + 240);
    ++*(_QWORD *)(*v9 + 224);
    if ( v13 )
    {
      v26 = 4 * v13;
      v14 = *(unsigned int *)(v12 + 248);
      if ( (unsigned int)(4 * v13) < 0x40 )
        v26 = 64;
      if ( v26 >= (unsigned int)v14 )
      {
LABEL_13:
        result = *(_QWORD *)(v12 + 232);
        for ( n = result + 16 * v14; n != result; result += 16 )
          *(_QWORD *)result = -4;
        *(_QWORD *)(v12 + 240) = 0;
        goto LABEL_16;
      }
      v27 = *(_QWORD **)(v12 + 232);
      v28 = v13 - 1;
      if ( v28 )
      {
        _BitScanReverse(&v28, v28);
        v29 = 1 << (33 - (v28 ^ 0x1F));
        if ( v29 < 64 )
          v29 = 64;
        if ( (_DWORD)v14 == v29 )
        {
          *(_QWORD *)(v12 + 240) = 0;
          result = (__int64)&v27[2 * (unsigned int)v14];
          do
          {
            if ( v27 )
              *v27 = -4;
            v27 += 2;
          }
          while ( (_QWORD *)result != v27 );
          goto LABEL_16;
        }
        v30 = (4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1);
        v31 = ((v30 | (v30 >> 2)) >> 4) | v30 | (v30 >> 2) | ((((v30 | (v30 >> 2)) >> 4) | v30 | (v30 >> 2)) >> 8);
        v67 = (v31 | (v31 >> 16)) + 1;
        v32 = 16 * ((v31 | (v31 >> 16)) + 1);
      }
      else
      {
        v67 = 128;
        v32 = 2048;
      }
      v70 = v32;
      j___libc_free_0(v27);
      *(_DWORD *)(v12 + 248) = v67;
      result = sub_22077B0(v70);
      v33 = *(unsigned int *)(v12 + 248);
      *(_QWORD *)(v12 + 240) = 0;
      *(_QWORD *)(v12 + 232) = result;
      for ( ii = result + 16 * v33; ii != result; result += 16 )
      {
        if ( result )
          *(_QWORD *)result = -4;
      }
    }
    else
    {
      result = *(unsigned int *)(v12 + 244);
      if ( (_DWORD)result )
      {
        v14 = *(unsigned int *)(v12 + 248);
        if ( (unsigned int)v14 <= 0x40 )
          goto LABEL_13;
        result = j___libc_free_0(*(_QWORD *)(v12 + 232));
        *(_QWORD *)(v12 + 232) = 0;
        *(_QWORD *)(v12 + 240) = 0;
        *(_DWORD *)(v12 + 248) = 0;
      }
    }
LABEL_16:
    *(_QWORD *)(v12 + 168) = 0;
    ++v9;
    *(_QWORD *)(v12 + 176) = 0;
    *(_QWORD *)(v12 + 184) = 0;
    *(_QWORD *)(v12 + 192) = 0;
    *(_QWORD *)(v12 + 200) = 0;
    *(_QWORD *)(v12 + 208) = 0;
  }
  if ( *(_DWORD *)(a1 + 208) )
  {
    result = *(_QWORD *)(a1 + 200);
    v35 = (__int64 *)(result + 16LL * *(unsigned int *)(a1 + 216));
    if ( (__int64 *)result != v35 )
    {
      while ( 1 )
      {
        v36 = *(_QWORD *)result;
        v37 = (__int64 *)result;
        if ( *(_QWORD *)result != -8 && v36 != -16 )
          break;
        result += 16;
        if ( v35 == (__int64 *)result )
          return result;
      }
      if ( v35 != (__int64 *)result )
      {
        v38 = *(_DWORD *)(a1 + 248);
        v39 = *(_QWORD *)(result + 8);
        v71 = a1 + 224;
        if ( !v38 )
          goto LABEL_64;
LABEL_54:
        v40 = *(_QWORD *)(a1 + 232);
        v41 = (v38 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        result = v40 + 112LL * v41;
        v42 = *(_QWORD *)result;
        if ( *(_QWORD *)result == v39 )
          goto LABEL_55;
        v56 = 1;
        v57 = 0;
        while ( v42 != -8 )
        {
          if ( v42 == -16 && !v57 )
            v57 = result;
          v41 = (v38 - 1) & (v56 + v41);
          result = v40 + 112LL * v41;
          v42 = *(_QWORD *)result;
          if ( *(_QWORD *)result == v39 )
          {
LABEL_55:
            v43 = *(_QWORD **)(result + 16);
            v44 = result + 8;
            if ( *(_QWORD **)(result + 24) == v43 )
            {
              v53 = &v43[*(unsigned int *)(result + 36)];
              v54 = *(_DWORD *)(result + 36);
              if ( v43 != v53 )
              {
                v55 = 0;
                while ( v36 != *v43 )
                {
                  if ( *v43 == -2 )
                    v55 = v43;
                  if ( ++v43 == v53 )
                  {
                    if ( v55 )
                    {
                      *v55 = v36;
                      --*(_DWORD *)(result + 40);
                      ++*(_QWORD *)(result + 8);
                      goto LABEL_57;
                    }
                    goto LABEL_89;
                  }
                }
                goto LABEL_57;
              }
              goto LABEL_89;
            }
LABEL_56:
            result = sub_16CCBA0(v44, v36);
            goto LABEL_57;
          }
          ++v56;
        }
        v58 = *(_DWORD *)(a1 + 240);
        if ( v57 )
          result = v57;
        ++*(_QWORD *)(a1 + 224);
        v49 = v58 + 1;
        if ( 4 * (v58 + 1) >= 3 * v38 )
        {
          while ( 1 )
          {
            sub_1616A10(v71, 2 * v38);
            v45 = *(_DWORD *)(a1 + 248);
            if ( !v45 )
              goto LABEL_121;
            v46 = v45 - 1;
            v47 = *(_QWORD *)(a1 + 232);
            v48 = v46 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            result = v47 + 112LL * v48;
            v49 = *(_DWORD *)(a1 + 240) + 1;
            v50 = *(_QWORD *)result;
            if ( *(_QWORD *)result != v39 )
              break;
LABEL_86:
            *(_DWORD *)(a1 + 240) = v49;
            if ( *(_QWORD *)result != -8 )
              --*(_DWORD *)(a1 + 244);
            v53 = (_QWORD *)(result + 48);
            *(_QWORD *)result = v39;
            v44 = result + 8;
            v54 = 0;
            *(_QWORD *)(result + 8) = 0;
            *(_QWORD *)(result + 16) = result + 48;
            *(_QWORD *)(result + 24) = result + 48;
            *(_QWORD *)(result + 32) = 8;
            *(_DWORD *)(result + 40) = 0;
LABEL_89:
            if ( *(_DWORD *)(result + 32) <= v54 )
              goto LABEL_56;
            *(_DWORD *)(result + 36) = v54 + 1;
            *v53 = v36;
            ++*(_QWORD *)(result + 8);
LABEL_57:
            v37 += 2;
            if ( v37 == v35 )
              return result;
            while ( 1 )
            {
              v36 = *v37;
              if ( *v37 != -16 && v36 != -8 )
                break;
              v37 += 2;
              if ( v35 == v37 )
                return result;
            }
            if ( v35 == v37 )
              return result;
            v38 = *(_DWORD *)(a1 + 248);
            v39 = v37[1];
            if ( v38 )
              goto LABEL_54;
LABEL_64:
            ++*(_QWORD *)(a1 + 224);
          }
          v51 = 1;
          v52 = 0;
          while ( v50 != -8 )
          {
            if ( v50 == -16 && !v52 )
              v52 = result;
            v48 = v46 & (v51 + v48);
            result = v47 + 112LL * v48;
            v50 = *(_QWORD *)result;
            if ( *(_QWORD *)result == v39 )
              goto LABEL_86;
            ++v51;
          }
        }
        else
        {
          if ( v38 - *(_DWORD *)(a1 + 244) - v49 > v38 >> 3 )
            goto LABEL_86;
          v68 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
          sub_1616A10(v71, v38);
          v59 = *(_DWORD *)(a1 + 248);
          if ( !v59 )
          {
LABEL_121:
            ++*(_DWORD *)(a1 + 240);
            BUG();
          }
          v60 = v59 - 1;
          v61 = *(_QWORD *)(a1 + 232);
          v52 = 0;
          v62 = 1;
          v63 = v60 & v68;
          result = v61 + 112LL * (v60 & v68);
          v49 = *(_DWORD *)(a1 + 240) + 1;
          v64 = *(_QWORD *)result;
          if ( *(_QWORD *)result == v39 )
            goto LABEL_86;
          while ( v64 != -8 )
          {
            if ( v64 == -16 && !v52 )
              v52 = result;
            v63 = v60 & (v62 + v63);
            result = v61 + 112LL * v63;
            v64 = *(_QWORD *)result;
            if ( *(_QWORD *)result == v39 )
              goto LABEL_86;
            ++v62;
          }
        }
        if ( v52 )
          result = v52;
        goto LABEL_86;
      }
    }
  }
  return result;
}
