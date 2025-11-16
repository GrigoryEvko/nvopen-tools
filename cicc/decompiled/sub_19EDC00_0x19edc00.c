// Function: sub_19EDC00
// Address: 0x19edc00
//
__int64 __fastcall sub_19EDC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int8 v9; // al
  unsigned __int64 v10; // rbx
  _QWORD *v11; // r15
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 result; // rax
  unsigned __int64 v16; // r8
  unsigned int v17; // r14d
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // r14
  __int64 v27; // rsi
  int v28; // eax
  int v29; // ecx
  __int64 v30; // r8
  unsigned int v31; // eax
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rsi
  int v37; // eax
  int v38; // r9d
  unsigned __int64 v39; // [rsp-40h] [rbp-40h]

  if ( !a4 )
    return 0;
  v9 = *(_BYTE *)(a4 + 16);
  if ( v9 <= 0x10u )
  {
    v10 = *(unsigned int *)(a2 + 32);
    v11 = *(_QWORD **)(a2 + 24);
    v12 = 0;
    if ( *(_DWORD *)(a2 + 32) )
    {
      if ( --v10 )
      {
        _BitScanReverse64(&v10, v10);
        v12 = 64 - (v10 ^ 0x3F);
        v10 = 8LL * v12;
      }
    }
    v13 = *(unsigned int *)(a1 + 176);
    if ( (unsigned int)v13 <= v12 )
    {
      v16 = v12 + 1;
      v17 = v12 + 1;
      if ( v16 < v13 )
      {
        *(_DWORD *)(a1 + 176) = v16;
      }
      else if ( v16 > v13 )
      {
        if ( v16 > *(unsigned int *)(a1 + 180) )
        {
          v39 = v12 + 1;
          sub_16CD150(a1 + 168, (const void *)(a1 + 184), v16, 8, v16, a6);
          v13 = *(unsigned int *)(a1 + 176);
          v16 = v39;
        }
        v14 = *(_QWORD *)(a1 + 168);
        v18 = (_QWORD *)(v14 + 8 * v13);
        v19 = (_QWORD *)(v14 + 8 * v16);
        if ( v18 != v19 )
        {
          do
          {
            if ( v18 )
              *v18 = 0;
            ++v18;
          }
          while ( v19 != v18 );
          v14 = *(_QWORD *)(a1 + 168);
        }
        *(_DWORD *)(a1 + 176) = v17;
        goto LABEL_8;
      }
    }
    v14 = *(_QWORD *)(a1 + 168);
LABEL_8:
    *v11 = *(_QWORD *)(v14 + v10);
    *(_QWORD *)(*(_QWORD *)(a1 + 168) + v10) = v11;
    return sub_19E59B0(a1, a4);
  }
  if ( v9 == 17 )
  {
    sub_19E1860(*(_QWORD **)(a2 + 24), *(_DWORD *)(a2 + 32), a1 + 168, a4, a5, a6);
    result = sub_145CDC0(0x20u, (__int64 *)(a1 + 64));
    if ( result )
    {
      *(_DWORD *)(result + 8) = 2;
      *(_QWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 24) = a4;
      *(_QWORD *)result = &unk_49F4CD0;
    }
    *(_DWORD *)(result + 12) = *(unsigned __int8 *)(a4 + 16);
    return result;
  }
  v20 = *(_DWORD *)(a1 + 1496);
  if ( !v20 )
    return 0;
  v21 = v20 - 1;
  v22 = *(_QWORD *)(a1 + 1480);
  v23 = (v20 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v24 = (__int64 *)(v22 + 16LL * (unsigned int)v23);
  v25 = *v24;
  if ( *v24 != a4 )
  {
    v37 = 1;
    while ( v25 != -8 )
    {
      a6 = v37 + 1;
      v23 = v21 & (unsigned int)(v37 + v23);
      v24 = (__int64 *)(v22 + 16LL * (unsigned int)v23);
      v25 = *v24;
      if ( a4 == *v24 )
        goto LABEL_22;
      v37 = a6;
    }
    return 0;
  }
LABEL_22:
  v26 = v24[1];
  if ( !v26 )
    return 0;
  v27 = *(_QWORD *)(v26 + 8);
  if ( a3 != v27 && v27 )
  {
    v28 = *(_DWORD *)(a1 + 1856);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 1840);
      v31 = (v28 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v32 = *(_QWORD *)(v30 + 8LL * v31);
      if ( a3 == v32 )
        return sub_19E5BE0(a1, v27);
      v38 = 1;
      while ( v32 != -8 )
      {
        v31 = v29 & (v38 + v31);
        v32 = *(_QWORD *)(v30 + 8LL * v31);
        if ( a3 == v32 )
          return sub_19E5BE0(a1, v27);
        ++v38;
      }
    }
    sub_19EDA30(a1, a4, a3);
    v27 = *(_QWORD *)(v26 + 8);
    return sub_19E5BE0(a1, v27);
  }
  result = *(_QWORD *)(v26 + 48);
  if ( result )
  {
    if ( a3 != a4 )
    {
      v33 = *(_DWORD *)(a1 + 1856);
      if ( !v33 )
      {
LABEL_39:
        sub_19EDA30(a1, a4, a3);
        goto LABEL_40;
      }
      v23 = (unsigned int)(v33 - 1);
      v34 = *(_QWORD *)(a1 + 1840);
      v35 = v23 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v36 = *(_QWORD *)(v34 + 8LL * v35);
      if ( a3 != v36 )
      {
        LODWORD(v25) = 1;
        while ( v36 != -8 )
        {
          a6 = v25 + 1;
          v35 = v23 & (v25 + v35);
          v36 = *(_QWORD *)(v34 + 8LL * v35);
          if ( a3 == v36 )
            goto LABEL_40;
          LODWORD(v25) = v25 + 1;
        }
        goto LABEL_39;
      }
    }
LABEL_40:
    sub_19E1860(*(_QWORD **)(a2 + 24), *(_DWORD *)(a2 + 32), a1 + 168, v23, v25, a6);
    return *(_QWORD *)(v26 + 48);
  }
  return result;
}
