// Function: sub_1472270
// Address: 0x1472270
//
__int64 __fastcall sub_1472270(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // eax
  __int64 *v10; // r14
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // rdx
  __int64 v15; // r14
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rcx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 *v25; // r9
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // rax
  int v29; // r10d
  __int64 *v30; // rdi
  int v31; // eax
  int v32; // edx
  int v33; // r15d
  __int64 *v34; // rdx
  int v35; // eax
  int v36; // edi
  __int64 v37; // r11
  __int64 v38[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v39[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1 + 624;
  v38[0] = a2;
  v6 = *(_DWORD *)(a1 + 648);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 624);
    goto LABEL_31;
  }
  v7 = v38[0];
  v8 = *(_QWORD *)(a1 + 632);
  v9 = (v6 - 1) & ((LODWORD(v38[0]) >> 9) ^ (LODWORD(v38[0]) >> 4));
  v10 = (__int64 *)(v8 + 56LL * v9);
  v11 = *v10;
  if ( v38[0] != *v10 )
  {
    v29 = 1;
    v30 = 0;
    while ( v11 != -8 )
    {
      if ( !v30 && v11 == -16 )
        v30 = v10;
      v9 = (v6 - 1) & (v29 + v9);
      v10 = (__int64 *)(v8 + 56LL * v9);
      v11 = *v10;
      if ( v38[0] == *v10 )
        goto LABEL_3;
      ++v29;
    }
    v31 = *(_DWORD *)(a1 + 640);
    if ( v30 )
      v10 = v30;
    ++*(_QWORD *)(a1 + 624);
    v32 = v31 + 1;
    if ( 4 * (v31 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 644) - v32 > v6 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a1 + 640) = v32;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 644);
        *v10 = v7;
        v10[1] = (__int64)(v10 + 3);
        v10[2] = 0x200000000LL;
        goto LABEL_12;
      }
LABEL_32:
      sub_146C520(v4, v6);
      sub_14612A0(v4, v38, v39);
      v10 = (__int64 *)v39[0];
      v7 = v38[0];
      v32 = *(_DWORD *)(a1 + 640) + 1;
      goto LABEL_27;
    }
LABEL_31:
    v6 *= 2;
    goto LABEL_32;
  }
LABEL_3:
  v12 = (_QWORD *)v10[1];
  v13 = *((unsigned int *)v10 + 4);
  v14 = &v12[2 * v13];
  if ( v14 == v12 )
  {
LABEL_10:
    if ( (unsigned int)v13 >= *((_DWORD *)v10 + 5) )
      sub_16CD150(v10 + 1, v10 + 3, 0, 16);
LABEL_12:
    v17 = *((unsigned int *)v10 + 4);
    v18 = (_QWORD *)(v10[1] + 16 * v17);
    if ( v18 )
    {
      *v18 = a3;
      v18[1] = 0;
      LODWORD(v17) = *((_DWORD *)v10 + 4);
    }
    v19 = v38[0];
    *((_DWORD *)v10 + 4) = v17 + 1;
    v20 = sub_1471910((_QWORD *)a1, v19, a3);
    v21 = *(_DWORD *)(a1 + 648);
    v15 = v20;
    if ( v21 )
    {
      v22 = v38[0];
      v23 = *(_QWORD *)(a1 + 632);
      v24 = (v21 - 1) & ((LODWORD(v38[0]) >> 9) ^ (LODWORD(v38[0]) >> 4));
      v25 = (__int64 *)(v23 + 56LL * v24);
      v26 = *v25;
      if ( *v25 == v38[0] )
      {
        v27 = v25[1];
        v28 = v27 + 16LL * *((unsigned int *)v25 + 4);
        goto LABEL_18;
      }
      v33 = 1;
      v34 = 0;
      while ( v26 != -8 )
      {
        if ( v34 || v26 != -16 )
          v25 = v34;
        v24 = (v21 - 1) & (v33 + v24);
        v37 = v23 + 56LL * v24;
        v26 = *(_QWORD *)v37;
        if ( v38[0] == *(_QWORD *)v37 )
        {
          v27 = *(_QWORD *)(v37 + 8);
          v28 = v27 + 16LL * *(unsigned int *)(v37 + 16);
LABEL_18:
          while ( v28 != v27 )
          {
            if ( *(_QWORD **)(v28 - 16) == a3 )
            {
              *(_QWORD *)(v28 - 8) = v15;
              return v15;
            }
            v28 -= 16;
          }
          return v15;
        }
        ++v33;
        v34 = v25;
        v25 = (__int64 *)(v23 + 56LL * v24);
      }
      v35 = *(_DWORD *)(a1 + 640);
      if ( !v34 )
        v34 = v25;
      ++*(_QWORD *)(a1 + 624);
      v36 = v35 + 1;
      if ( 4 * (v35 + 1) >= 3 * v21 )
        goto LABEL_43;
      if ( v21 - *(_DWORD *)(a1 + 644) - v36 <= v21 >> 3 )
        goto LABEL_44;
    }
    else
    {
      ++*(_QWORD *)(a1 + 624);
LABEL_43:
      v21 *= 2;
LABEL_44:
      sub_146C520(v4, v21);
      sub_14612A0(v4, v38, v39);
      v34 = (__int64 *)v39[0];
      v22 = v38[0];
      v36 = *(_DWORD *)(a1 + 640) + 1;
    }
    *(_DWORD *)(a1 + 640) = v36;
    if ( *v34 != -8 )
      --*(_DWORD *)(a1 + 644);
    *v34 = v22;
    v34[1] = (__int64)(v34 + 3);
    v34[2] = 0x200000000LL;
    return v15;
  }
  while ( (_QWORD *)*v12 != a3 )
  {
    v12 += 2;
    if ( v14 == v12 )
      goto LABEL_10;
  }
  v15 = v12[1];
  if ( !v15 )
    return v38[0];
  return v15;
}
