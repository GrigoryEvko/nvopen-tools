// Function: sub_D2F240
// Address: 0xd2f240
//
__int64 __fastcall sub_D2F240(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // edi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r11d
  __int64 *v16; // r8
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r10
  __int64 *v20; // rax
  __int64 v21; // r12
  _QWORD *v22; // rdi
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 result; // rax
  _QWORD *v27; // rax
  int v28; // r8d
  __int64 *v29; // r10
  const void *v30; // r9
  __int64 v31; // rcx
  int v32; // edx
  __int64 *v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // edx
  __int64 v39; // rcx
  int v40; // eax
  __int64 v41; // rsi
  unsigned int v42; // r8d
  int v43; // r9d
  int v44; // esi
  int v45; // r9d
  int v46; // r8d
  __int64 *v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h] BYREF
  __int64 v50[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a2 + 8);
  sub_D23E90(a2, a3);
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 120);
  v9 = *(_QWORD *)(*a1 + 104);
  if ( !v8 )
  {
    v49 = a3;
    v14 = v7 + 96;
    goto LABEL_40;
  }
  v10 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
  {
LABEL_3:
    *v12 = -8192;
    --*(_DWORD *)(v7 + 112);
    ++*(_DWORD *)(v7 + 116);
    v7 = *a1;
    v49 = a3;
    v8 = *(_DWORD *)(v7 + 120);
    v9 = *(_QWORD *)(v7 + 104);
    v14 = v7 + 96;
    v10 = v8 - 1;
    if ( v8 )
      goto LABEL_4;
LABEL_40:
    v50[0] = 0;
    v8 = 0;
    ++*(_QWORD *)(v7 + 96);
LABEL_41:
    v8 *= 2;
    goto LABEL_42;
  }
  v40 = 1;
  while ( v13 != -4096 )
  {
    v45 = v40 + 1;
    v11 = v10 & (v40 + v11);
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
      goto LABEL_3;
    v40 = v45;
  }
  v49 = a3;
  v14 = v7 + 96;
LABEL_4:
  v15 = 1;
  v16 = 0;
  v17 = v10 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v9 + 16LL * v17);
  v19 = *v18;
  if ( a3 == *v18 )
  {
LABEL_5:
    v20 = v18 + 1;
    goto LABEL_6;
  }
  while ( v19 != -4096 )
  {
    if ( v19 == -8192 && !v16 )
      v16 = v18;
    v17 = v10 & (v15 + v17);
    v18 = (__int64 *)(v9 + 16LL * v17);
    v19 = *v18;
    if ( a3 == *v18 )
      goto LABEL_5;
    ++v15;
  }
  if ( !v16 )
    v16 = v18;
  v50[0] = (__int64)v16;
  v37 = *(_DWORD *)(v7 + 112);
  ++*(_QWORD *)(v7 + 96);
  v38 = v37 + 1;
  if ( 4 * (v37 + 1) >= 3 * v8 )
    goto LABEL_41;
  v39 = a3;
  if ( v8 - *(_DWORD *)(v7 + 116) - v38 <= v8 >> 3 )
  {
LABEL_42:
    v48 = v14;
    sub_D25040(v14, v8);
    sub_D24A00(v48, &v49, v50);
    v39 = v49;
    v16 = (__int64 *)v50[0];
    v38 = *(_DWORD *)(v7 + 112) + 1;
  }
  *(_DWORD *)(v7 + 112) = v38;
  if ( *v16 != -4096 )
    --*(_DWORD *)(v7 + 116);
  *v16 = v39;
  v20 = v16 + 1;
  v16[1] = 0;
LABEL_6:
  *v20 = a2;
  v21 = *a1;
  if ( *(_DWORD *)(*a1 + 624) )
  {
    result = *(unsigned int *)(v21 + 632);
    v31 = *(_QWORD *)(v21 + 616);
    if ( (_DWORD)result )
    {
      v32 = result - 1;
      result = ((_DWORD)result - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v33 = (__int64 *)(v31 + 8 * result);
      v34 = *v33;
      if ( v6 == *v33 )
      {
        v50[0] = v6;
LABEL_24:
        *v33 = -8192;
        v35 = *(unsigned int *)(v21 + 648);
        --*(_DWORD *)(v21 + 624);
        v36 = *(_QWORD **)(v21 + 640);
        ++*(_DWORD *)(v21 + 628);
        v24 = (__int64)&v36[v35];
        v27 = sub_D22DD0(v36, v24, v50);
        v30 = v27 + 1;
        if ( v27 + 1 == (_QWORD *)v24 )
          goto LABEL_18;
        goto LABEL_17;
      }
      v41 = *v33;
      v42 = result;
      v43 = 1;
      while ( v41 != -4096 )
      {
        v42 = v32 & (v43 + v42);
        v41 = *(_QWORD *)(v31 + 8LL * v42);
        if ( v6 == v41 )
        {
          v50[0] = v6;
          v44 = 1;
          while ( v34 != -4096 )
          {
            v46 = v44 + 1;
            LODWORD(result) = v32 & (v44 + result);
            v33 = (__int64 *)(v31 + 8LL * (unsigned int)result);
            v34 = *v33;
            if ( v6 == *v33 )
              goto LABEL_24;
            v44 = v46;
          }
          v29 = v50;
          goto LABEL_19;
        }
        ++v43;
      }
    }
    return result;
  }
  v22 = *(_QWORD **)(v21 + 640);
  v23 = *(unsigned int *)(v21 + 648);
  v24 = (__int64)&v22[v23];
  v25 = (8 * v23) >> 3;
  if ( (8 * v23) >> 5 )
  {
    result = *(_QWORD *)(v21 + 640);
    while ( v6 != *(_QWORD *)result )
    {
      if ( v6 == *(_QWORD *)(result + 8) )
      {
        result += 8;
        break;
      }
      if ( v6 == *(_QWORD *)(result + 16) )
      {
        result += 16;
        break;
      }
      if ( v6 == *(_QWORD *)(result + 24) )
      {
        result += 24;
        break;
      }
      result += 32;
      if ( &v22[4 * ((8 * v23) >> 5)] == (_QWORD *)result )
      {
        v25 = (v24 - result) >> 3;
        goto LABEL_47;
      }
    }
LABEL_14:
    if ( v24 == result )
      return result;
    v50[0] = v6;
    v27 = sub_D22DD0(v22, v24, v50);
    if ( (_QWORD *)v24 == v27 )
      goto LABEL_19;
    v30 = v27 + 1;
    if ( (_QWORD *)v24 == v27 + 1 )
    {
LABEL_18:
      *(_DWORD *)(v21 + 648) = v28 - 1;
      v21 = *a1;
LABEL_19:
      v50[0] = a3;
      return sub_D2EA30(v21 + 608, v29);
    }
LABEL_17:
    v47 = v29;
    memmove(v27, v30, v24 - (_QWORD)v30);
    v28 = *(_DWORD *)(v21 + 648);
    v29 = v47;
    goto LABEL_18;
  }
  result = *(_QWORD *)(v21 + 640);
LABEL_47:
  if ( v25 != 2 )
  {
    if ( v25 != 3 )
    {
      if ( v25 != 1 )
        return result;
      goto LABEL_50;
    }
    if ( v6 == *(_QWORD *)result )
      goto LABEL_14;
    result += 8;
  }
  if ( v6 == *(_QWORD *)result )
    goto LABEL_14;
  result += 8;
LABEL_50:
  if ( v6 == *(_QWORD *)result )
    goto LABEL_14;
  return result;
}
