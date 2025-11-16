// Function: sub_33669C0
// Address: 0x33669c0
//
__int64 __fastcall sub_33669C0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v8; // r9
  char v9; // dl
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // r8
  unsigned int v14; // r10d
  unsigned int v15; // r11d
  unsigned __int64 v16; // rsi
  __int64 v17; // r14
  unsigned __int64 v18; // rdx
  int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rdi
  unsigned __int64 v23; // r9
  _QWORD *v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r12
  unsigned __int64 v27; // rax
  int v28; // edx
  unsigned __int64 v29; // rcx
  __int64 v30; // r12
  unsigned __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // rax
  unsigned int v37; // [rsp+4h] [rbp-6Ch]
  unsigned int v38; // [rsp+8h] [rbp-68h]
  unsigned int v39; // [rsp+Ch] [rbp-64h]
  __int64 v40; // [rsp+10h] [rbp-60h]
  unsigned __int64 v41; // [rsp+18h] [rbp-58h]
  int v42; // [rsp+20h] [rbp-50h]
  unsigned int v43; // [rsp+24h] [rbp-4Ch]
  unsigned __int64 v44; // [rsp+28h] [rbp-48h]
  unsigned int v45; // [rsp+30h] [rbp-40h]
  __int64 v47; // [rsp+38h] [rbp-38h]

  v47 = a2;
  v5 = sub_B2E500(*a1);
  result = sub_B2A630(v5);
  v42 = result;
  if ( !a2 )
    return result;
  v45 = result - 7;
  v43 = result - 9;
LABEL_3:
  for ( result = sub_AA4FF0(v47); ; result = sub_AA4FF0(v21) )
  {
    if ( !result )
      BUG();
    v9 = *(_BYTE *)(result - 24);
    if ( v9 == 95 )
      break;
    if ( v9 == 80 )
    {
      v29 = *(unsigned int *)(a4 + 12);
      v30 = *(_QWORD *)(a1[7] + 8LL * *(unsigned int *)(v47 + 44));
      v31 = *(unsigned int *)(a4 + 8);
      v32 = *(_DWORD *)(a4 + 8);
      if ( v31 >= v29 )
      {
        if ( v29 < v31 + 1 )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v31 + 1, 0x10u, a3, v31 + 1);
          v31 = *(unsigned int *)(a4 + 8);
        }
        v36 = (_QWORD *)(*(_QWORD *)a4 + 16 * v31);
        *v36 = v30;
        v36[1] = a3;
        v33 = *(_QWORD *)a4;
        v35 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v35;
      }
      else
      {
        v33 = *(_QWORD *)a4;
        v34 = *(_QWORD *)a4 + 16 * v31;
        if ( v34 )
        {
          *(_QWORD *)v34 = v30;
          *(_DWORD *)(v34 + 8) = a3;
          v32 = *(_DWORD *)(a4 + 8);
          v33 = *(_QWORD *)a4;
        }
        v35 = (unsigned int)(v32 + 1);
        *(_DWORD *)(a4 + 8) = v35;
      }
      result = *(_QWORD *)(v33 + 16 * v35 - 16);
      *(_BYTE *)(result + 233) = 1;
      if ( v42 != 12 )
      {
        result = *(_QWORD *)(*(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - 16);
        *(_BYTE *)(result + 235) = 1;
      }
      return result;
    }
    if ( v9 != 39 )
      goto LABEL_3;
    v10 = *(_QWORD *)(result - 32);
    v11 = v10 + 32LL * (*(_DWORD *)(result - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(result - 22) & 1) != 0 )
    {
      v12 = v10 + 64;
      if ( v11 == v10 + 64 )
        goto LABEL_22;
    }
    else
    {
      v12 = v10 + 32;
      if ( v11 == v10 + 32 )
        return result;
    }
    v13 = a3;
    v14 = v45;
    v15 = v43;
    do
    {
      v16 = *(unsigned int *)(a4 + 12);
      v17 = *(_QWORD *)(a1[7] + 8LL * *(unsigned int *)(*(_QWORD *)v12 + 44LL));
      v18 = *(unsigned int *)(a4 + 8);
      v19 = *(_DWORD *)(a4 + 8);
      if ( v18 >= v16 )
      {
        v23 = v44 & 0xFFFFFFFF00000000LL | (unsigned int)v13;
        v44 = v23;
        if ( v16 < v18 + 1 )
        {
          v37 = v15;
          v38 = v14;
          v39 = v13;
          v40 = result;
          v41 = v23;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v18 + 1, 0x10u, v13, v23);
          v18 = *(unsigned int *)(a4 + 8);
          v15 = v37;
          v14 = v38;
          v13 = v39;
          result = v40;
          v23 = v41;
        }
        v24 = (_QWORD *)(*(_QWORD *)a4 + 16 * v18);
        *v24 = v17;
        v24[1] = v23;
        ++*(_DWORD *)(a4 + 8);
      }
      else
      {
        v20 = *(_QWORD *)a4 + 16 * v18;
        if ( v20 )
        {
          *(_QWORD *)v20 = v17;
          *(_DWORD *)(v20 + 8) = v13;
          v19 = *(_DWORD *)(a4 + 8);
        }
        *(_DWORD *)(a4 + 8) = v19 + 1;
      }
      if ( v15 <= 1 )
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - 16) + 235LL) = 1;
      if ( v14 > 1 )
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - 16) + 233LL) = 1;
      v12 += 32;
    }
    while ( v11 != v12 );
    if ( (*(_BYTE *)(result - 22) & 1) == 0 )
      return result;
    v10 = *(_QWORD *)(result - 32);
LABEL_22:
    v21 = *(_QWORD *)(v10 + 32);
    if ( !v21 )
      return result;
    v22 = a1[4];
    if ( v22 )
      a3 = ((unsigned int)sub_FF0430(v22, v47, *(_QWORD *)(v10 + 32)) * (unsigned __int64)a3 + 0x40000000) >> 31;
    v47 = v21;
  }
  v25 = *(unsigned int *)(a4 + 12);
  v26 = *(_QWORD *)(a1[7] + 8LL * *(unsigned int *)(v47 + 44));
  v27 = *(unsigned int *)(a4 + 8);
  v28 = *(_DWORD *)(a4 + 8);
  if ( v27 >= v25 )
  {
    if ( v25 < v27 + 1 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v27 + 1, 0x10u, a3, v8);
      v27 = *(unsigned int *)(a4 + 8);
    }
    result = *(_QWORD *)a4 + 16 * v27;
    *(_QWORD *)result = v26;
    *(_QWORD *)(result + 8) = a3;
    ++*(_DWORD *)(a4 + 8);
  }
  else
  {
    result = *(_QWORD *)a4 + 16 * v27;
    if ( result )
    {
      *(_QWORD *)result = v26;
      *(_DWORD *)(result + 8) = a3;
      v28 = *(_DWORD *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = v28 + 1;
  }
  return result;
}
