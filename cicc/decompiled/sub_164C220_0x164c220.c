// Function: sub_164C220
// Address: 0x164c220
//
unsigned __int64 __fastcall sub_164C220(__int64 a1)
{
  __int64 *v2; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v5; // r8d
  unsigned __int64 v6; // r13
  __int64 v7; // rdi
  unsigned int v8; // edx
  _QWORD *v9; // rsi
  __int64 v10; // rax
  _QWORD *v11; // rdx
  unsigned __int64 result; // rax
  unsigned int v13; // esi
  __int64 v14; // rdi
  __int64 v15; // r8
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rcx
  __int64 *v20; // rsi
  unsigned __int64 v21; // rdi
  int v22; // r8d
  int v23; // r8d
  __int64 v24; // r9
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // r10d
  _QWORD *v29; // r9
  int v30; // eax
  int v31; // edi
  int v32; // edi
  __int64 v33; // r8
  _QWORD *v34; // r9
  __int64 v35; // r15
  int v36; // eax
  __int64 v37; // rcx
  int v38; // r11d
  _QWORD *v39; // r10
  int v40; // ecx
  int v41; // edx
  int v42; // r8d
  int v43; // r8d
  int v44; // esi
  __int64 v45; // r9
  _QWORD *v46; // rcx
  unsigned int v47; // edx
  __int64 v48; // rdi
  int v49; // edi
  int v50; // edi
  __int64 v51; // r8
  __int64 v52; // r13
  __int64 v53; // rcx
  int v54; // edx
  _QWORD *v55; // rsi
  int v56; // ecx
  _QWORD *v57; // r10

  v2 = (__int64 *)sub_16498A0(*(_QWORD *)(a1 + 16));
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *v2;
  if ( (*(_BYTE *)(v3 + 17) & 1) != 0 )
  {
    v13 = *(_DWORD *)(v4 + 2664);
    v14 = v4 + 2640;
    if ( v13 )
    {
      v15 = *(_QWORD *)(v4 + 2648);
      v16 = (v13 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v17 = (_QWORD *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v3 == *v17 )
        return sub_1649AC0((unsigned __int64 *)a1, (unsigned __int64)(v17 + 1));
      v38 = 1;
      v39 = 0;
      while ( v18 != -8 )
      {
        if ( !v39 && v18 == -16 )
          v39 = v17;
        v16 = (v13 - 1) & (v38 + v16);
        v17 = (_QWORD *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v3 == *v17 )
          return sub_1649AC0((unsigned __int64 *)a1, (unsigned __int64)(v17 + 1));
        ++v38;
      }
      v40 = *(_DWORD *)(v4 + 2656);
      if ( v39 )
        v17 = v39;
      ++*(_QWORD *)(v4 + 2640);
      v41 = v40 + 1;
      if ( 4 * (v40 + 1) < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(v4 + 2660) - v41 > v13 >> 3 )
        {
LABEL_46:
          *(_DWORD *)(v4 + 2656) = v41;
          if ( *v17 != -8 )
            --*(_DWORD *)(v4 + 2660);
          *v17 = v3;
          v17[1] = 0;
          return sub_1649AC0((unsigned __int64 *)a1, (unsigned __int64)(v17 + 1));
        }
        sub_164B930(v14, v13);
        v49 = *(_DWORD *)(v4 + 2664);
        if ( !v49 )
          goto LABEL_97;
        v50 = v49 - 1;
        v51 = *(_QWORD *)(v4 + 2648);
        LODWORD(v52) = v50 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v17 = (_QWORD *)(v51 + 16LL * (unsigned int)v52);
        v53 = *v17;
        if ( v3 != *v17 )
        {
          v54 = 1;
          v55 = 0;
          while ( v53 != -8 )
          {
            if ( v53 == -16 && !v55 )
              v55 = v17;
            v52 = v50 & (unsigned int)(v52 + v54);
            v17 = (_QWORD *)(v51 + 16 * v52);
            v53 = *v17;
            if ( v3 == *v17 )
              goto LABEL_52;
            ++v54;
          }
          if ( v55 )
          {
            v41 = *(_DWORD *)(v4 + 2656) + 1;
            v17 = v55;
            goto LABEL_46;
          }
        }
LABEL_52:
        v41 = *(_DWORD *)(v4 + 2656) + 1;
        goto LABEL_46;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 2640);
    }
    sub_164B930(v14, 2 * v13);
    v42 = *(_DWORD *)(v4 + 2664);
    if ( !v42 )
      goto LABEL_97;
    v43 = v42 - 1;
    v44 = 1;
    v45 = *(_QWORD *)(v4 + 2648);
    v46 = 0;
    v47 = v43 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v17 = (_QWORD *)(v45 + 16LL * v47);
    v48 = *v17;
    if ( v3 != *v17 )
    {
      while ( v48 != -8 )
      {
        if ( !v46 && v48 == -16 )
          v46 = v17;
        v47 = v43 & (v44 + v47);
        v17 = (_QWORD *)(v45 + 16LL * v47);
        v48 = *v17;
        if ( v3 == *v17 )
          goto LABEL_52;
        ++v44;
      }
      if ( v46 )
      {
        v41 = *(_DWORD *)(v4 + 2656) + 1;
        v17 = v46;
        goto LABEL_46;
      }
    }
    goto LABEL_52;
  }
  v5 = *(_DWORD *)(v4 + 2664);
  v6 = *(_QWORD *)(v4 + 2648);
  v7 = v4 + 2640;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 2640);
    goto LABEL_23;
  }
  v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v9 = (_QWORD *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == v3 )
    goto LABEL_4;
  v28 = 1;
  v29 = 0;
  while ( v10 != -8 )
  {
    if ( !v29 && v10 == -16 )
      v29 = v9;
    v8 = (v5 - 1) & (v28 + v8);
    v9 = (_QWORD *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v3 == *v9 )
      goto LABEL_4;
    ++v28;
  }
  v30 = *(_DWORD *)(v4 + 2656);
  if ( v29 )
    v9 = v29;
  ++*(_QWORD *)(v4 + 2640);
  v25 = v30 + 1;
  if ( 4 * (v30 + 1) >= 3 * v5 )
  {
LABEL_23:
    sub_164B930(v7, 2 * v5);
    v22 = *(_DWORD *)(v4 + 2664);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v4 + 2648);
      v25 = *(_DWORD *)(v4 + 2656) + 1;
      v26 = v23 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v9 = (_QWORD *)(v24 + 16LL * v26);
      v27 = *v9;
      if ( v3 != *v9 )
      {
        v56 = 1;
        v57 = 0;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !v57 )
            v57 = v9;
          v26 = v23 & (v56 + v26);
          v9 = (_QWORD *)(v24 + 16LL * v26);
          v27 = *v9;
          if ( v3 == *v9 )
            goto LABEL_25;
          ++v56;
        }
        if ( v57 )
          v9 = v57;
      }
      goto LABEL_25;
    }
    goto LABEL_97;
  }
  if ( v5 - *(_DWORD *)(v4 + 2660) - v25 <= v5 >> 3 )
  {
    sub_164B930(v7, v5);
    v31 = *(_DWORD *)(v4 + 2664);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(v4 + 2648);
      v34 = 0;
      LODWORD(v35) = v32 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v25 = *(_DWORD *)(v4 + 2656) + 1;
      v36 = 1;
      v9 = (_QWORD *)(v33 + 16LL * (unsigned int)v35);
      v37 = *v9;
      if ( v3 != *v9 )
      {
        while ( v37 != -8 )
        {
          if ( !v34 && v37 == -16 )
            v34 = v9;
          v35 = v32 & (unsigned int)(v35 + v36);
          v9 = (_QWORD *)(v33 + 16 * v35);
          v37 = *v9;
          if ( v3 == *v9 )
            goto LABEL_25;
          ++v36;
        }
        if ( v34 )
          v9 = v34;
      }
      goto LABEL_25;
    }
LABEL_97:
    ++*(_DWORD *)(v4 + 2656);
    BUG();
  }
LABEL_25:
  *(_DWORD *)(v4 + 2656) = v25;
  if ( *v9 != -8 )
    --*(_DWORD *)(v4 + 2660);
  *v9 = v3;
  v9[1] = 0;
LABEL_4:
  sub_1649AC0((unsigned __int64 *)a1, (unsigned __int64)(v9 + 1));
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 17LL) |= 1u;
  v11 = *(_QWORD **)(v4 + 2648);
  if ( (unsigned __int64)v11 > v6 || (result = (unsigned __int64)&v11[2 * *(unsigned int *)(v4 + 2664)], v6 >= result) )
  {
    result = *(unsigned int *)(v4 + 2656);
    if ( (unsigned int)result > 1 )
    {
      v19 = &v11[2 * *(unsigned int *)(v4 + 2664)];
      if ( v11 != v19 )
      {
        while ( 1 )
        {
          result = (unsigned __int64)v11;
          if ( *v11 != -16 && *v11 != -8 )
            break;
          v11 += 2;
          if ( v19 == v11 )
            return result;
        }
        while ( (_QWORD *)result != v19 )
        {
          v20 = *(__int64 **)(result + 8);
          v21 = result + 8;
          result += 16LL;
          *v20 = v21 | *v20 & 7;
          if ( (_QWORD *)result == v19 )
            break;
          while ( *(_QWORD *)result == -16 || *(_QWORD *)result == -8 )
          {
            result += 16LL;
            if ( v19 == (_QWORD *)result )
              return result;
          }
        }
      }
    }
  }
  return result;
}
