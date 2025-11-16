// Function: sub_2E2DA00
// Address: 0x2e2da00
//
unsigned __int64 __fastcall sub_2E2DA00(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v9; // rcx
  int v10; // ebx
  _DWORD *v11; // rsi
  __int64 v12; // rdi
  _DWORD *v13; // rdx
  _DWORD *v14; // rbx
  _DWORD *v15; // r13
  __int64 v16; // r9
  _DWORD *v17; // rcx
  int v18; // edx
  unsigned int v19; // esi
  int v20; // eax
  int v21; // esi
  __int64 v22; // r11
  unsigned int v23; // eax
  _DWORD *v24; // r8
  int v25; // ecx
  int v26; // edx
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // edi
  int v30; // r13d
  _DWORD *v31; // r10
  unsigned int v32; // edx
  _DWORD *v33; // rcx
  int v34; // esi
  int v35; // eax
  int v36; // ebx
  int v37; // r11d
  int v38; // eax
  int v39; // eax
  int v40; // esi
  __int64 v41; // r11
  _DWORD *v42; // r9
  int v43; // edi
  int v44; // ecx
  int v45; // eax
  int v46; // r9d
  __int64 v47; // r11
  unsigned int v48; // edx
  int v49; // esi
  int v50; // edi
  _DWORD *v51; // rcx
  int v52; // r9d
  __int64 v53; // r11
  int v54; // edi
  unsigned int v55; // edx
  int v56; // esi
  int v57; // r10d
  int v58; // r14d
  __int64 v59; // rax

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v27 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v27 )
    {
      v28 = *(_QWORD *)(a1 + 8);
      v29 = *a2;
      v30 = 1;
      v31 = 0;
      v32 = (v27 - 1) & (37 * *a2);
      v33 = (_DWORD *)(v28 + 4LL * v32);
      v34 = *v33;
      if ( v29 == *v33 )
        return result;
      while ( v34 != 0x7FFFFFFF )
      {
        if ( v31 || v34 != 0x80000000 )
          v33 = v31;
        v32 = (v27 - 1) & (v30 + v32);
        v34 = *(_DWORD *)(v28 + 4LL * v32);
        if ( v29 == v34 )
          return result;
        ++v30;
        v31 = v33;
        v33 = (_DWORD *)(v28 + 4LL * v32);
      }
      if ( !v31 )
        v31 = v33;
      v35 = result + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v35 < (unsigned int)(3 * v27) )
      {
        if ( (int)v27 - *(_DWORD *)(a1 + 20) - v35 > (unsigned int)v27 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 16) = v35;
          if ( *v31 != 0x7FFFFFFF )
            --*(_DWORD *)(a1 + 20);
          v36 = *a2;
          *v31 = v36;
          result = *(unsigned int *)(a1 + 40);
          if ( result + 1 > *(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 4u, v27, v28);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * result) = v36;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
        sub_29F8760(a1, v27);
        v52 = *(_DWORD *)(a1 + 24);
        if ( v52 )
        {
          v27 = *a2;
          v28 = (unsigned int)(v52 - 1);
          v53 = *(_QWORD *)(a1 + 8);
          v51 = 0;
          v54 = 1;
          v55 = v28 & (37 * v27);
          v31 = (_DWORD *)(v53 + 4LL * v55);
          v56 = *v31;
          v35 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v31 == (_DWORD)v27 )
            goto LABEL_35;
          while ( v56 != 0x7FFFFFFF )
          {
            if ( !v51 && v56 == 0x80000000 )
              v51 = v31;
            v55 = v28 & (v54 + v55);
            v31 = (_DWORD *)(v53 + 4LL * v55);
            v56 = *v31;
            if ( (_DWORD)v27 == *v31 )
              goto LABEL_35;
            ++v54;
          }
          goto LABEL_69;
        }
        goto LABEL_108;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_29F8760(a1, 2 * v27);
    v46 = *(_DWORD *)(a1 + 24);
    if ( v46 )
    {
      v27 = *a2;
      v28 = (unsigned int)(v46 - 1);
      v47 = *(_QWORD *)(a1 + 8);
      v48 = v28 & (37 * v27);
      v31 = (_DWORD *)(v47 + 4LL * v48);
      v49 = *v31;
      v35 = *(_DWORD *)(a1 + 16) + 1;
      if ( (_DWORD)v27 == *v31 )
        goto LABEL_35;
      v50 = 1;
      v51 = 0;
      while ( v49 != 0x7FFFFFFF )
      {
        if ( v49 == 0x80000000 && !v51 )
          v51 = v31;
        v48 = v28 & (v50 + v48);
        v31 = (_DWORD *)(v47 + 4LL * v48);
        v49 = *v31;
        if ( (_DWORD)v27 == *v31 )
          goto LABEL_35;
        ++v50;
      }
LABEL_69:
      if ( v51 )
        v31 = v51;
      goto LABEL_35;
    }
LABEL_108:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  v9 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v10 = *a2;
  v11 = (_DWORD *)(result + 4 * v9);
  v12 = (4 * v9) >> 2;
  if ( !((4 * v9) >> 4) )
    goto LABEL_12;
  v13 = (_DWORD *)(result + 16 * ((4 * v9) >> 4));
  do
  {
    if ( *(_DWORD *)result == v10 )
      goto LABEL_9;
    if ( *(_DWORD *)(result + 4) == v10 )
    {
      result += 4LL;
      if ( v11 == (_DWORD *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_DWORD *)(result + 8) == v10 )
    {
      result += 8LL;
      if ( v11 == (_DWORD *)result )
        goto LABEL_15;
      return result;
    }
    if ( *(_DWORD *)(result + 12) == v10 )
    {
      result += 12LL;
      if ( v11 == (_DWORD *)result )
        goto LABEL_15;
      return result;
    }
    result += 16LL;
  }
  while ( v13 != (_DWORD *)result );
  v12 = (__int64)((__int64)v11 - result) >> 2;
LABEL_12:
  if ( v12 == 2 )
  {
LABEL_42:
    if ( *(_DWORD *)result != v10 )
    {
      result += 4LL;
      goto LABEL_44;
    }
LABEL_9:
    if ( v11 == (_DWORD *)result )
      goto LABEL_15;
    return result;
  }
  if ( v12 == 3 )
  {
    if ( *(_DWORD *)result == v10 )
      goto LABEL_9;
    result += 4LL;
    goto LABEL_42;
  }
  if ( v12 != 1 )
    goto LABEL_15;
LABEL_44:
  if ( *(_DWORD *)result == v10 )
    goto LABEL_9;
LABEL_15:
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v9 + 1, 4u, a5, a6);
    v11 = (_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40));
  }
  *v11 = v10;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 8 )
  {
    v14 = *(_DWORD **)(a1 + 32);
    v15 = &v14[result];
    while ( 1 )
    {
      v19 = *(_DWORD *)(a1 + 24);
      if ( !v19 )
        break;
      v16 = *(_QWORD *)(a1 + 8);
      result = (v19 - 1) & (37 * *v14);
      v17 = (_DWORD *)(v16 + 4 * result);
      v18 = *v17;
      if ( *v14 != *v17 )
      {
        v37 = 1;
        v24 = 0;
        while ( v18 != 0x7FFFFFFF )
        {
          if ( v24 || v18 != 0x80000000 )
            v17 = v24;
          result = (v19 - 1) & (v37 + (_DWORD)result);
          v18 = *(_DWORD *)(v16 + 4LL * (unsigned int)result);
          if ( *v14 == v18 )
            goto LABEL_20;
          ++v37;
          v24 = v17;
          v17 = (_DWORD *)(v16 + 4LL * (unsigned int)result);
        }
        v38 = *(_DWORD *)(a1 + 16);
        if ( !v24 )
          v24 = v17;
        ++*(_QWORD *)a1;
        v26 = v38 + 1;
        if ( 4 * (v38 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 20) - v26 <= v19 >> 3 )
          {
            sub_29F8760(a1, v19);
            v39 = *(_DWORD *)(a1 + 24);
            if ( !v39 )
            {
LABEL_107:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v40 = v39 - 1;
            v41 = *(_QWORD *)(a1 + 8);
            v42 = 0;
            v24 = (_DWORD *)(v41 + 4LL * ((v39 - 1) & (unsigned int)(37 * *v14)));
            v43 = (v39 - 1) & (37 * *v14);
            v44 = *v24;
            v26 = *(_DWORD *)(a1 + 16) + 1;
            v45 = 1;
            if ( *v14 != *v24 )
            {
              while ( v44 != 0x7FFFFFFF )
              {
                if ( !v42 && v44 == 0x80000000 )
                  v42 = v24;
                v58 = v45 + 1;
                v59 = v40 & (unsigned int)(v43 + v45);
                v24 = (_DWORD *)(v41 + 4 * v59);
                v43 = v59;
                v44 = *v24;
                if ( *v14 == *v24 )
                  goto LABEL_25;
                v45 = v58;
              }
LABEL_61:
              if ( v42 )
                v24 = v42;
            }
          }
LABEL_25:
          *(_DWORD *)(a1 + 16) = v26;
          if ( *v24 != 0x7FFFFFFF )
            --*(_DWORD *)(a1 + 20);
          result = (unsigned int)*v14;
          *v24 = result;
          goto LABEL_20;
        }
LABEL_23:
        sub_29F8760(a1, 2 * v19);
        v20 = *(_DWORD *)(a1 + 24);
        if ( !v20 )
          goto LABEL_107;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a1 + 8);
        v23 = (v20 - 1) & (37 * *v14);
        v24 = (_DWORD *)(v22 + 4LL * (v21 & (unsigned int)(37 * *v14)));
        v25 = *v24;
        v26 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v14 != *v24 )
        {
          v57 = 1;
          v42 = 0;
          while ( v25 != 0x7FFFFFFF )
          {
            if ( v25 == 0x80000000 && !v42 )
              v42 = v24;
            v23 = v21 & (v57 + v23);
            v24 = (_DWORD *)(v22 + 4LL * v23);
            v25 = *v24;
            if ( *v14 == *v24 )
              goto LABEL_25;
            ++v57;
          }
          goto LABEL_61;
        }
        goto LABEL_25;
      }
LABEL_20:
      if ( v15 == ++v14 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  return result;
}
