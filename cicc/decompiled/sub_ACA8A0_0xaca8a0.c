// Function: sub_ACA8A0
// Address: 0xaca8a0
//
__int64 __fastcall sub_ACA8A0(__int64 **a1)
{
  __int64 v2; // rbx
  unsigned int v3; // esi
  __int64 v4; // rdi
  int v5; // r14d
  __int64 v6; // r8
  __int64 ***v7; // rdx
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  __int64 **v10; // r10
  __int64 *v11; // rbx
  __int64 result; // rax
  int v13; // eax
  int v14; // ecx
  __int64 v15; // r12
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 **v20; // rdi
  int v21; // r10d
  __int64 ***v22; // r9
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  __int64 ***v26; // r8
  unsigned int v27; // r13d
  int v28; // r9d
  __int64 **v29; // rsi
  __int64 v30; // [rsp+8h] [rbp-28h]

  v2 = **a1;
  v3 = *(_DWORD *)(v2 + 1928);
  v4 = v2 + 1904;
  if ( !v3 )
  {
    ++*(_QWORD *)(v2 + 1904);
    goto LABEL_23;
  }
  v5 = 1;
  v6 = *(_QWORD *)(v2 + 1912);
  v7 = 0;
  v8 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v9 = (_QWORD *)(v6 + 16LL * v8);
  v10 = (__int64 **)*v9;
  if ( (__int64 **)*v9 == a1 )
  {
LABEL_3:
    v11 = v9 + 1;
    result = v9[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v10 != (__int64 **)-4096LL )
  {
    if ( !v7 && v10 == (__int64 **)-8192LL )
      v7 = (__int64 ***)v9;
    v8 = (v3 - 1) & (v5 + v8);
    v9 = (_QWORD *)(v6 + 16LL * v8);
    v10 = (__int64 **)*v9;
    if ( (__int64 **)*v9 == a1 )
      goto LABEL_3;
    ++v5;
  }
  if ( !v7 )
    v7 = (__int64 ***)v9;
  v13 = *(_DWORD *)(v2 + 1920);
  ++*(_QWORD *)(v2 + 1904);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v3 )
  {
LABEL_23:
    sub_ACA690(v4, 2 * v3);
    v16 = *(_DWORD *)(v2 + 1928);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v2 + 1912);
      v19 = (v16 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = *(_DWORD *)(v2 + 1920) + 1;
      v7 = (__int64 ***)(v18 + 16LL * v19);
      v20 = *v7;
      if ( *v7 != a1 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != (__int64 **)-4096LL )
        {
          if ( !v22 && v20 == (__int64 **)-8192LL )
            v22 = v7;
          v19 = v17 & (v21 + v19);
          v7 = (__int64 ***)(v18 + 16LL * v19);
          v20 = *v7;
          if ( *v7 == a1 )
            goto LABEL_15;
          ++v21;
        }
        if ( v22 )
          v7 = v22;
      }
      goto LABEL_15;
    }
    goto LABEL_46;
  }
  if ( v3 - *(_DWORD *)(v2 + 1924) - v14 <= v3 >> 3 )
  {
    sub_ACA690(v4, v3);
    v23 = *(_DWORD *)(v2 + 1928);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v2 + 1912);
      v26 = 0;
      v27 = v24 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v28 = 1;
      v14 = *(_DWORD *)(v2 + 1920) + 1;
      v7 = (__int64 ***)(v25 + 16LL * v27);
      v29 = *v7;
      if ( *v7 != a1 )
      {
        while ( v29 != (__int64 **)-4096LL )
        {
          if ( !v26 && v29 == (__int64 **)-8192LL )
            v26 = v7;
          v27 = v24 & (v28 + v27);
          v7 = (__int64 ***)(v25 + 16LL * v27);
          v29 = *v7;
          if ( *v7 == a1 )
            goto LABEL_15;
          ++v28;
        }
        if ( v26 )
          v7 = v26;
      }
      goto LABEL_15;
    }
LABEL_46:
    ++*(_DWORD *)(v2 + 1920);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v2 + 1920) = v14;
  if ( *v7 != (__int64 **)-4096LL )
    --*(_DWORD *)(v2 + 1924);
  *v7 = a1;
  v11 = (__int64 *)(v7 + 1);
  v7[1] = 0;
LABEL_18:
  result = sub_BD2C40(24, unk_3F289A4);
  if ( result )
  {
    v30 = result;
    sub_BD35F0(result, a1, 12);
    result = v30;
    *(_DWORD *)(v30 + 4) &= 0x38000000u;
  }
  v15 = *v11;
  *v11 = result;
  if ( v15 )
  {
    sub_BD7260(v15);
    sub_BD2DD0(v15);
    return *v11;
  }
  return result;
}
