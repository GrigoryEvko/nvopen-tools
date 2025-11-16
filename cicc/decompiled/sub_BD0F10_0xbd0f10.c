// Function: sub_BD0F10
// Address: 0xbd0f10
//
__int64 __fastcall sub_BD0F10(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  __int64 v5; // r8
  int v6; // r10d
  unsigned int v7; // edi
  __int64 *v8; // rdx
  __int64 result; // rax
  __int64 v10; // rcx
  int v11; // edi
  int v12; // ecx
  _QWORD *v13; // rdi
  __int64 v14; // rbx
  bool v15; // zf
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rdi
  int v19; // r10d
  _QWORD *v20; // r9
  unsigned int v21; // ecx
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r12
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  int v32; // eax
  int v33; // eax
  int v34; // eax
  int v35; // r8d
  unsigned int v36; // r14d
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  int v39; // r10d
  _QWORD *v40; // r8
  int v41; // r9d
  int v42; // r9d
  __int64 v43; // r10
  unsigned int v44; // edx
  __int64 v45; // rdi
  int v46; // r8d
  int v47; // esi
  __int64 v48; // r9
  int v49; // r8d
  unsigned int v50; // r12d
  __int64 v51; // rdx
  __int64 v52; // rdi
  __int64 v53; // [rsp+28h] [rbp-78h]
  __int64 v54; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v55; // [rsp+40h] [rbp-60h] BYREF
  __int64 v56; // [rsp+48h] [rbp-58h]
  _QWORD v57[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = *(unsigned int *)(a1 + 120);
  v53 = a1 + 96;
  if ( !(_DWORD)v4 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_63;
  }
  v5 = *(_QWORD *)(a1 + 104);
  v6 = 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 8LL * v7);
  result = 0;
  v10 = *v8;
  if ( *v8 == a2 )
    return result;
  while ( v10 != -4096 )
  {
    if ( !result && v10 == -8192 )
      result = (__int64)v8;
    v7 = (v4 - 1) & (v6 + v7);
    v8 = (__int64 *)(v5 + 8LL * v7);
    v10 = *v8;
    if ( *v8 == a2 )
      return result;
    ++v6;
  }
  v11 = *(_DWORD *)(a1 + 112);
  if ( !result )
    result = (__int64)v8;
  ++*(_QWORD *)(a1 + 96);
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= (unsigned int)(3 * v4) )
  {
LABEL_63:
    sub_BCFDB0(v53, 2 * v4);
    v41 = *(_DWORD *)(a1 + 120);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(a1 + 104);
      v4 = *(unsigned int *)(a1 + 112);
      v44 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = v4 + 1;
      result = v43 + 8LL * v44;
      v45 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        v46 = 1;
        v4 = 0;
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v4 )
            v4 = result;
          v44 = v42 & (v46 + v44);
          result = v43 + 8LL * v44;
          v45 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_14;
          ++v46;
        }
        if ( v4 )
          result = v4;
      }
      goto LABEL_14;
    }
    goto LABEL_97;
  }
  if ( (int)v4 - *(_DWORD *)(a1 + 116) - v12 <= (unsigned int)v4 >> 3 )
  {
    sub_BCFDB0(v53, v4);
    v47 = *(_DWORD *)(a1 + 120);
    if ( v47 )
    {
      v4 = (unsigned int)(v47 - 1);
      v48 = *(_QWORD *)(a1 + 104);
      v49 = 1;
      v50 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = *(_DWORD *)(a1 + 112) + 1;
      v51 = 0;
      result = v48 + 8LL * v50;
      v52 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        while ( v52 != -4096 )
        {
          if ( !v51 && v52 == -8192 )
            v51 = result;
          v50 = v4 & (v49 + v50);
          result = v48 + 8LL * v50;
          v52 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_14;
          ++v49;
        }
        if ( v51 )
          result = v51;
      }
      goto LABEL_14;
    }
LABEL_97:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 112) = v12;
  if ( *(_QWORD *)result != -4096 )
    --*(_DWORD *)(a1 + 116);
  v13 = v57;
  *(_QWORD *)result = a2;
  v55 = v57;
  v57[0] = a2;
  v56 = 0x400000001LL;
  LODWORD(result) = 1;
  do
  {
    v14 = v13[(unsigned int)result - 1];
    LODWORD(v56) = result - 1;
    if ( *(_BYTE *)(v14 + 8) == 15 )
    {
      v15 = *(_BYTE *)(a1 + 152) == 0;
      v54 = v14;
      if ( v15 || *(_QWORD *)(v14 + 24) )
      {
        v4 = *(_QWORD *)(a1 + 136);
        if ( v4 == *(_QWORD *)(a1 + 144) )
        {
          sub_9CABF0(a1 + 128, (_BYTE *)v4, &v54);
        }
        else
        {
          if ( v4 )
          {
            *(_QWORD *)v4 = v14;
            v4 = *(_QWORD *)(a1 + 136);
          }
          v4 += 8LL;
          *(_QWORD *)(a1 + 136) = v4;
        }
      }
    }
    v16 = *(_QWORD *)(v14 + 16);
    v17 = v16 + 8LL * *(unsigned int *)(v14 + 12);
    if ( v17 != v16 )
    {
      while ( 1 )
      {
        v4 = *(unsigned int *)(a1 + 120);
        v24 = *(_QWORD *)(v17 - 8);
        if ( !(_DWORD)v4 )
          break;
        v18 = *(_QWORD *)(a1 + 104);
        v19 = 1;
        v20 = 0;
        v21 = (v4 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v22 = (_QWORD *)(v18 + 8LL * v21);
        v23 = *v22;
        if ( v24 == *v22 )
        {
LABEL_27:
          v17 -= 8;
          if ( v16 == v17 )
            goto LABEL_37;
        }
        else
        {
          while ( v23 != -4096 )
          {
            if ( v20 || v23 != -8192 )
              v22 = v20;
            v21 = (v4 - 1) & (v19 + v21);
            v23 = *(_QWORD *)(v18 + 8LL * v21);
            if ( v24 == v23 )
              goto LABEL_27;
            ++v19;
            v20 = v22;
            v22 = (_QWORD *)(v18 + 8LL * v21);
          }
          v32 = *(_DWORD *)(a1 + 112);
          if ( !v20 )
            v20 = v22;
          ++*(_QWORD *)(a1 + 96);
          v29 = v32 + 1;
          if ( 4 * (v32 + 1) < (unsigned int)(3 * v4) )
          {
            if ( (int)v4 - *(_DWORD *)(a1 + 116) - v29 <= (unsigned int)v4 >> 3 )
            {
              sub_BCFDB0(v53, v4);
              v33 = *(_DWORD *)(a1 + 120);
              if ( !v33 )
              {
LABEL_96:
                ++*(_DWORD *)(a1 + 112);
                BUG();
              }
              v34 = v33 - 1;
              v4 = *(_QWORD *)(a1 + 104);
              v35 = 1;
              v36 = v34 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v20 = (_QWORD *)(v4 + 8LL * v36);
              v29 = *(_DWORD *)(a1 + 112) + 1;
              v37 = 0;
              v38 = *v20;
              if ( v24 != *v20 )
              {
                while ( v38 != -4096 )
                {
                  if ( !v37 && v38 == -8192 )
                    v37 = v20;
                  v36 = v34 & (v35 + v36);
                  v20 = (_QWORD *)(v4 + 8LL * v36);
                  v38 = *v20;
                  if ( v24 == *v20 )
                    goto LABEL_32;
                  ++v35;
                }
                if ( v37 )
                  v20 = v37;
              }
            }
            goto LABEL_32;
          }
LABEL_30:
          sub_BCFDB0(v53, 2 * v4);
          v25 = *(_DWORD *)(a1 + 120);
          if ( !v25 )
            goto LABEL_96;
          v26 = v25 - 1;
          v27 = *(_QWORD *)(a1 + 104);
          v28 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v20 = (_QWORD *)(v27 + 8LL * v28);
          v4 = *v20;
          v29 = *(_DWORD *)(a1 + 112) + 1;
          if ( v24 != *v20 )
          {
            v39 = 1;
            v40 = 0;
            while ( v4 != -4096 )
            {
              if ( !v40 && v4 == -8192 )
                v40 = v20;
              v28 = v26 & (v39 + v28);
              v20 = (_QWORD *)(v27 + 8LL * v28);
              v4 = *v20;
              if ( v24 == *v20 )
                goto LABEL_32;
              ++v39;
            }
            if ( v40 )
              v20 = v40;
          }
LABEL_32:
          *(_DWORD *)(a1 + 112) = v29;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a1 + 116);
          *v20 = v24;
          v30 = (unsigned int)v56;
          v31 = (unsigned int)v56 + 1LL;
          if ( v31 > HIDWORD(v56) )
          {
            v4 = (unsigned __int64)v57;
            sub_C8D5F0(&v55, v57, v31, 8);
            v30 = (unsigned int)v56;
          }
          v17 -= 8;
          v55[v30] = v24;
          LODWORD(v56) = v56 + 1;
          if ( v16 == v17 )
            goto LABEL_37;
        }
      }
      ++*(_QWORD *)(a1 + 96);
      goto LABEL_30;
    }
LABEL_37:
    result = (unsigned int)v56;
    v13 = v55;
  }
  while ( (_DWORD)v56 );
  if ( v55 != v57 )
    return _libc_free(v55, v4);
  return result;
}
