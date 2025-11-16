// Function: sub_16176C0
// Address: 0x16176c0
//
_QWORD *__fastcall sub_16176C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  unsigned int v5; // r8d
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rsi
  _QWORD *result; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r8
  unsigned int v17; // edi
  __int64 v18; // rcx
  unsigned int v19; // esi
  __int64 v20; // r13
  int v21; // edi
  int v22; // edi
  __int64 v23; // r10
  unsigned int v24; // esi
  int v25; // ecx
  __int64 v26; // r8
  _QWORD *v27; // r11
  int v28; // ecx
  int v29; // esi
  int v30; // esi
  _QWORD *v31; // r10
  unsigned int v32; // r14d
  __int64 v33; // r8
  int v34; // r11d
  __int64 v35; // rdi
  int v36; // r11d
  __int64 *v37; // r10
  int v38; // edx
  int v39; // esi
  int v40; // r14d
  _QWORD *v41; // r11
  int v42; // esi
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+8h] [rbp-58h]
  _QWORD *v45; // [rsp+10h] [rbp-50h]
  int v46; // [rsp+10h] [rbp-50h]
  _QWORD *v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  __int64 v49; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v50; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_DWORD *)(a1 + 248);
  v48 = a1 + 224;
  v49 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 224);
LABEL_44:
    v42 = 2 * v5;
LABEL_45:
    sub_1617500(v48, v42);
    sub_16122E0(v48, &v49, &v50);
    v8 = v50;
    v4 = v49;
    v39 = *(_DWORD *)(a1 + 240) + 1;
    goto LABEL_35;
  }
  v6 = *(_QWORD *)(a1 + 232);
  v7 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( v4 == *v8 )
    goto LABEL_3;
  v36 = 1;
  v37 = 0;
  while ( v9 != -4 )
  {
    if ( v9 == -8 && !v37 )
      v37 = v8;
    v7 = (v5 - 1) & (v36 + v7);
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
      goto LABEL_3;
    ++v36;
  }
  v38 = *(_DWORD *)(a1 + 240);
  if ( v37 )
    v8 = v37;
  ++*(_QWORD *)(a1 + 224);
  v39 = v38 + 1;
  if ( 4 * (v38 + 1) >= 3 * v5 )
    goto LABEL_44;
  if ( v5 - *(_DWORD *)(a1 + 244) - v39 <= v5 >> 3 )
  {
    v42 = v5;
    goto LABEL_45;
  }
LABEL_35:
  *(_DWORD *)(a1 + 240) = v39;
  if ( *v8 != -4 )
    --*(_DWORD *)(a1 + 244);
  *v8 = v4;
  v9 = v49;
  v8[1] = 0;
LABEL_3:
  v8[1] = a2;
  result = (_QWORD *)sub_1614F20(*(_QWORD *)(a1 + 16), v9);
  v11 = result;
  if ( result )
  {
    result = (_QWORD *)result[6];
    v12 = (__int64)(v11[7] - (_QWORD)result) >> 3;
    if ( (_DWORD)v12 )
    {
      v13 = a2;
      v14 = 0;
      v15 = 8LL * (unsigned int)(v12 - 1);
      while ( 1 )
      {
        v19 = *(_DWORD *)(a1 + 248);
        v20 = *(_QWORD *)(result[v14 / 8] + 32LL);
        if ( !v19 )
          break;
        v16 = *(_QWORD *)(a1 + 232);
        v17 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        result = (_QWORD *)(v16 + 16LL * v17);
        v18 = *result;
        if ( v20 == *result )
        {
LABEL_7:
          result[1] = v13;
          if ( v14 == v15 )
            return result;
          goto LABEL_8;
        }
        v46 = 1;
        v27 = 0;
        while ( v18 != -4 )
        {
          if ( !v27 && v18 == -8 )
            v27 = result;
          v17 = (v19 - 1) & (v46 + v17);
          result = (_QWORD *)(v16 + 16LL * v17);
          v18 = *result;
          if ( v20 == *result )
            goto LABEL_7;
          ++v46;
        }
        v28 = *(_DWORD *)(a1 + 240);
        if ( v27 )
          result = v27;
        ++*(_QWORD *)(a1 + 224);
        v25 = v28 + 1;
        if ( 4 * v25 >= 3 * v19 )
          goto LABEL_11;
        if ( v19 - *(_DWORD *)(a1 + 244) - v25 <= v19 >> 3 )
        {
          v44 = v13;
          v47 = v11;
          sub_1617500(v48, v19);
          v29 = *(_DWORD *)(a1 + 248);
          if ( !v29 )
          {
LABEL_67:
            ++*(_DWORD *)(a1 + 240);
            BUG();
          }
          v30 = v29 - 1;
          v31 = 0;
          v13 = v44;
          v32 = v30 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v33 = *(_QWORD *)(a1 + 232);
          v34 = 1;
          v25 = *(_DWORD *)(a1 + 240) + 1;
          v11 = v47;
          result = (_QWORD *)(v33 + 16LL * v32);
          v35 = *result;
          if ( v20 != *result )
          {
            while ( v35 != -4 )
            {
              if ( !v31 && v35 == -8 )
                v31 = result;
              v32 = v30 & (v34 + v32);
              result = (_QWORD *)(v33 + 16LL * v32);
              v35 = *result;
              if ( v20 == *result )
                goto LABEL_13;
              ++v34;
            }
            if ( v31 )
              result = v31;
          }
        }
LABEL_13:
        *(_DWORD *)(a1 + 240) = v25;
        if ( *result != -4 )
          --*(_DWORD *)(a1 + 244);
        result[1] = 0;
        *result = v20;
        result[1] = v13;
        if ( v14 == v15 )
          return result;
LABEL_8:
        result = (_QWORD *)v11[6];
        v14 += 8LL;
      }
      ++*(_QWORD *)(a1 + 224);
LABEL_11:
      v43 = v13;
      v45 = v11;
      sub_1617500(v48, 2 * v19);
      v21 = *(_DWORD *)(a1 + 248);
      if ( !v21 )
        goto LABEL_67;
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 232);
      v13 = v43;
      v24 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v25 = *(_DWORD *)(a1 + 240) + 1;
      v11 = v45;
      result = (_QWORD *)(v23 + 16LL * v24);
      v26 = *result;
      if ( v20 != *result )
      {
        v40 = 1;
        v41 = 0;
        while ( v26 != -4 )
        {
          if ( !v41 && v26 == -8 )
            v41 = result;
          v24 = v22 & (v40 + v24);
          result = (_QWORD *)(v23 + 16LL * v24);
          v26 = *result;
          if ( v20 == *result )
            goto LABEL_13;
          ++v40;
        }
        if ( v41 )
          result = v41;
      }
      goto LABEL_13;
    }
  }
  return result;
}
