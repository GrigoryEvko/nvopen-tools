// Function: sub_31420E0
// Address: 0x31420e0
//
__int64 __fastcall sub_31420E0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r10d
  unsigned int v10; // edi
  __int64 v11; // rcx
  __int64 v12; // rdx
  int v13; // esi
  __int64 v14; // r8
  int v15; // esi
  unsigned __int64 v16; // r11
  int v17; // r15d
  _QWORD *v18; // rdx
  __int64 v19; // r11
  unsigned int v20; // eax
  int v21; // ecx
  int v22; // edx
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rsi
  int v28; // r10d
  __int64 v29; // r9
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // r8
  int v33; // r10d
  __int64 v34; // rdi
  __int64 v35; // rsi
  unsigned int v36; // [rsp+4h] [rbp-3Ch]
  __int64 v37; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 16);
  result = v3 + 32LL * *(unsigned int *)(a1 + 24) - 32;
  v37 = result;
  if ( v3 != result )
  {
    v36 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
    while ( 1 )
    {
      v7 = *(_DWORD *)(v3 + 24);
      if ( !v7 )
        break;
      v8 = *(_QWORD *)(v3 + 8);
      v9 = 1;
      v10 = (v7 - 1) & v36;
      v11 = v8 + 72LL * v10;
      result = 0;
      v12 = *(_QWORD *)v11;
      if ( *(_QWORD *)v11 != a3 )
      {
        while ( v12 != -4096 )
        {
          if ( v12 == -8192 && !result )
            result = v11;
          v10 = (v7 - 1) & (v9 + v10);
          v11 = v8 + 72LL * v10;
          v12 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 == a3 )
            goto LABEL_5;
          ++v9;
        }
        if ( !result )
          result = v11;
        v21 = *(_DWORD *)(v3 + 16);
        ++*(_QWORD *)v3;
        v22 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(v3 + 20) - v22 <= v7 >> 3 )
          {
            sub_22EBA60(v3, v7);
            v30 = *(_DWORD *)(v3 + 24);
            if ( !v30 )
            {
LABEL_53:
              ++*(_DWORD *)(v3 + 16);
              BUG();
            }
            v31 = v30 - 1;
            v32 = *(_QWORD *)(v3 + 8);
            v33 = 1;
            v29 = 0;
            LODWORD(v34) = v31 & v36;
            v22 = *(_DWORD *)(v3 + 16) + 1;
            result = v32 + 72LL * (v31 & v36);
            v35 = *(_QWORD *)result;
            if ( *(_QWORD *)result != a3 )
            {
              while ( v35 != -4096 )
              {
                if ( !v29 && v35 == -8192 )
                  v29 = result;
                v34 = v31 & (unsigned int)(v34 + v33);
                result = v32 + 72 * v34;
                v35 = *(_QWORD *)result;
                if ( *(_QWORD *)result == a3 )
                  goto LABEL_20;
                ++v33;
              }
              goto LABEL_37;
            }
          }
          goto LABEL_20;
        }
LABEL_33:
        sub_22EBA60(v3, 2 * v7);
        v23 = *(_DWORD *)(v3 + 24);
        if ( !v23 )
          goto LABEL_53;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(v3 + 8);
        LODWORD(v26) = v24 & v36;
        v22 = *(_DWORD *)(v3 + 16) + 1;
        result = v25 + 72LL * (v24 & v36);
        v27 = *(_QWORD *)result;
        if ( *(_QWORD *)result != a3 )
        {
          v28 = 1;
          v29 = 0;
          while ( v27 != -4096 )
          {
            if ( !v29 && v27 == -8192 )
              v29 = result;
            v26 = v24 & (unsigned int)(v26 + v28);
            result = v25 + 72 * v26;
            v27 = *(_QWORD *)result;
            if ( *(_QWORD *)result == a3 )
              goto LABEL_20;
            ++v28;
          }
LABEL_37:
          if ( v29 )
            result = v29;
        }
LABEL_20:
        *(_DWORD *)(v3 + 16) = v22;
        if ( *(_QWORD *)result != -4096 )
          --*(_DWORD *)(v3 + 20);
        *(_QWORD *)result = a3;
        *(_OWORD *)(result + 8) = 0;
        *(_OWORD *)(result + 24) = 0;
        *(_OWORD *)(result + 40) = 0;
        *(_OWORD *)(result + 56) = 0;
        goto LABEL_23;
      }
LABEL_5:
      v13 = *(_DWORD *)(v11 + 32);
      if ( v13 )
      {
        v14 = a2[1];
        v15 = v13 - 1;
        v16 = (0xBF58476D1CE4E5B9LL
             * ((969526130LL * (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4))) & 0xFFFFFFFELL
              | ((unsigned __int64)(((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)) << 32))) >> 31;
        v17 = 1;
        for ( result = v15
                     & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                      * ((unsigned int)v16
                                       ^ (-279380126 * (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)))
                                       | ((unsigned __int64)(((unsigned int)a2[2] >> 9) ^ ((unsigned int)a2[2] >> 4)) << 32))) >> 31)
                      ^ (484763065
                       * ((unsigned int)v16 ^ (-279380126 * (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4))))));
              ;
              result = v15 & v20 )
        {
          v18 = (_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL * (unsigned int)result);
          v19 = v18[2];
          if ( v19 == a2[2] && v14 == v18[1] && *a2 == *v18 )
            break;
          if ( v19 == -4096 && v18[1] == -4096 && *v18 == -4096 )
            goto LABEL_23;
          v20 = v17 + result;
          ++v17;
        }
        v18[2] = -8192;
        v3 += 32;
        v18[1] = -8192;
        *v18 = -8192;
        --*(_DWORD *)(v11 + 24);
        ++*(_DWORD *)(v11 + 28);
        if ( v37 == v3 )
          return result;
      }
      else
      {
LABEL_23:
        v3 += 32;
        if ( v37 == v3 )
          return result;
      }
    }
    ++*(_QWORD *)v3;
    goto LABEL_33;
  }
  return result;
}
