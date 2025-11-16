// Function: sub_322C540
// Address: 0x322c540
//
__int64 __fastcall sub_322C540(_QWORD *a1, __int64 a2, int a3)
{
  __int64 v5; // r8
  char v6; // cl
  __int64 v7; // rdi
  int v8; // eax
  unsigned int v9; // edx
  __int64 *v10; // rbx
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rbx
  char v18; // dl
  __int64 v19; // rcx
  int v20; // esi
  _QWORD *v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rbx
  unsigned int v24; // esi
  unsigned int v25; // eax
  _QWORD *v26; // r9
  int v27; // ecx
  unsigned int v28; // edi
  __int64 v29; // rbx
  int v30; // r10d
  int v31; // r10d
  __int64 v32; // rsi
  int v33; // edx
  unsigned int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // rsi
  int v37; // edx
  unsigned int v38; // eax
  __int64 v39; // rcx
  int v40; // r8d
  _QWORD *v41; // rdi
  int v42; // edx
  int v43; // edx
  int v44; // r8d

  v5 = *a1;
  v6 = *(_BYTE *)(*a1 + 8LL) & 1;
  if ( v6 )
  {
    v7 = v5 + 16;
    v8 = 3;
  }
  else
  {
    v23 = *(unsigned int *)(v5 + 24);
    v7 = *(_QWORD *)(v5 + 16);
    if ( !(_DWORD)v23 )
      goto LABEL_24;
    v8 = v23 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_4;
  v30 = 1;
  while ( v11 != -4096 )
  {
    v9 = v8 & (v30 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_4;
    ++v30;
  }
  if ( v6 )
  {
    v29 = 64;
    goto LABEL_25;
  }
  v23 = *(unsigned int *)(v5 + 24);
LABEL_24:
  v29 = 16 * v23;
LABEL_25:
  v10 = (__int64 *)(v7 + v29);
LABEL_4:
  v12 = 64;
  if ( !v6 )
    v12 = 16LL * *(unsigned int *)(v5 + 24);
  result = v7 + v12;
  if ( v10 != (__int64 *)result )
  {
    v14 = v10[1];
    result = sub_B10CD0(v14 + 56);
    if ( *(_DWORD *)(result + 4) != a3 )
    {
      v15 = *a1;
      *v10 = -8192;
      v16 = *(_DWORD *)(v15 + 8);
      ++*(_DWORD *)(v15 + 12);
      *(_DWORD *)(v15 + 8) = (2 * (v16 >> 1) - 2) | v16 & 1;
      v17 = a1[1];
      v18 = *(_BYTE *)(v17 + 3008) & 1;
      if ( v18 )
      {
        v19 = v17 + 3016;
        v20 = 3;
      }
      else
      {
        v24 = *(_DWORD *)(v17 + 3024);
        v19 = *(_QWORD *)(v17 + 3016);
        if ( !v24 )
        {
          v25 = *(_DWORD *)(v17 + 3008);
          ++*(_QWORD *)(v17 + 3000);
          v26 = 0;
          v27 = (v25 >> 1) + 1;
          goto LABEL_17;
        }
        v20 = v24 - 1;
      }
      result = v20 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v21 = (_QWORD *)(v19 + 8 * result);
      v22 = *v21;
      if ( v14 == *v21 )
        return result;
      v31 = 1;
      v26 = 0;
      while ( v22 != -4096 )
      {
        if ( v22 != -8192 || v26 )
          v21 = v26;
        result = v20 & (unsigned int)(v31 + result);
        v22 = *(_QWORD *)(v19 + 8LL * (unsigned int)result);
        if ( v14 == v22 )
          return result;
        ++v31;
        v26 = v21;
        v21 = (_QWORD *)(v19 + 8LL * (unsigned int)result);
      }
      v25 = *(_DWORD *)(v17 + 3008);
      if ( !v26 )
        v26 = v21;
      ++*(_QWORD *)(v17 + 3000);
      v27 = (v25 >> 1) + 1;
      if ( v18 )
      {
        v28 = 12;
        v24 = 4;
LABEL_18:
        if ( 4 * v27 < v28 )
        {
          if ( v24 - *(_DWORD *)(v17 + 3012) - v27 > v24 >> 3 )
          {
LABEL_20:
            result = (2 * (v25 >> 1) + 2) | v25 & 1;
            *(_DWORD *)(v17 + 3008) = result;
            if ( *v26 != -4096 )
              --*(_DWORD *)(v17 + 3012);
            *v26 = v14;
            return result;
          }
          sub_322C130(v17 + 3000, v24);
          if ( (*(_BYTE *)(v17 + 3008) & 1) != 0 )
          {
            v36 = v17 + 3016;
            v37 = 3;
            goto LABEL_42;
          }
          v43 = *(_DWORD *)(v17 + 3024);
          v36 = *(_QWORD *)(v17 + 3016);
          if ( v43 )
          {
            v37 = v43 - 1;
LABEL_42:
            v38 = v37 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v26 = (_QWORD *)(v36 + 8LL * v38);
            v39 = *v26;
            if ( v14 != *v26 )
            {
              v40 = 1;
              v41 = 0;
              while ( v39 != -4096 )
              {
                if ( !v41 && v39 == -8192 )
                  v41 = v26;
                v38 = v37 & (v40 + v38);
                v26 = (_QWORD *)(v36 + 8LL * v38);
                v39 = *v26;
                if ( v14 == *v26 )
                  goto LABEL_39;
                ++v40;
              }
LABEL_45:
              if ( v41 )
                v26 = v41;
              goto LABEL_39;
            }
            goto LABEL_39;
          }
LABEL_71:
          *(_DWORD *)(v17 + 3008) = (2 * (*(_DWORD *)(v17 + 3008) >> 1) + 2) | *(_DWORD *)(v17 + 3008) & 1;
          BUG();
        }
        sub_322C130(v17 + 3000, 2 * v24);
        if ( (*(_BYTE *)(v17 + 3008) & 1) != 0 )
        {
          v32 = v17 + 3016;
          v33 = 3;
        }
        else
        {
          v42 = *(_DWORD *)(v17 + 3024);
          v32 = *(_QWORD *)(v17 + 3016);
          if ( !v42 )
            goto LABEL_71;
          v33 = v42 - 1;
        }
        v34 = v33 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v26 = (_QWORD *)(v32 + 8LL * v34);
        v35 = *v26;
        if ( v14 != *v26 )
        {
          v44 = 1;
          v41 = 0;
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v41 )
              v41 = v26;
            v34 = v33 & (v44 + v34);
            v26 = (_QWORD *)(v32 + 8LL * v34);
            v35 = *v26;
            if ( v14 == *v26 )
              goto LABEL_39;
            ++v44;
          }
          goto LABEL_45;
        }
LABEL_39:
        v25 = *(_DWORD *)(v17 + 3008);
        goto LABEL_20;
      }
      v24 = *(_DWORD *)(v17 + 3024);
LABEL_17:
      v28 = 3 * v24;
      goto LABEL_18;
    }
  }
  return result;
}
