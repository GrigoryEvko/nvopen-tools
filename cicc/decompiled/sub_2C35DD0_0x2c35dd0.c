// Function: sub_2C35DD0
// Address: 0x2c35dd0
//
__int64 __fastcall sub_2C35DD0(__int64 a1, int a2)
{
  __int64 v2; // r12
  int v4; // eax
  __int64 v5; // rdx
  size_t v6; // rdx
  void *v7; // rdi
  int *v8; // rsi
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // ecx
  unsigned int v13; // eax
  int v14; // r13d
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r9
  unsigned int v18; // esi
  int v19; // eax
  _DWORD *v20; // rdx
  int v21; // eax
  __int64 v22; // r8
  int v23; // r12d
  int v24; // r13d
  __int64 v25; // r14
  int *v26; // r15
  unsigned int v27; // esi
  int v28; // eax
  _DWORD *v29; // rdx
  int v30; // eax
  int v31; // [rsp+Ch] [rbp-44h] BYREF
  _DWORD *v32; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1 + 112;
  v4 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v31 = a2;
  if ( v4 )
  {
    v12 = 4 * v4;
    v5 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)(4 * v4) < 0x40 )
      v12 = 64;
    if ( v12 >= (unsigned int)v5 )
      goto LABEL_4;
    v13 = v4 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v13, v13);
      v14 = 1 << (33 - (v13 ^ 0x1F));
      if ( v14 < 64 )
        v14 = 64;
      if ( v14 == (_DWORD)v5 )
        goto LABEL_19;
    }
    else
    {
      v14 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 120), 4 * v5, 4);
    v15 = sub_2C261E0(v14);
    *(_DWORD *)(a1 + 136) = v15;
    if ( !v15 )
      goto LABEL_40;
    *(_QWORD *)(a1 + 120) = sub_C7D670(4LL * v15, 4);
LABEL_19:
    sub_2C2BFC0(v2);
    v16 = *(_DWORD *)(a1 + 128);
    *(_DWORD *)(a1 + 152) = 0;
    if ( !v16 )
      goto LABEL_8;
    result = sub_22B31A0(v2, &v31, &v32);
    if ( (_BYTE)result )
      return result;
    v18 = *(_DWORD *)(a1 + 136);
    v19 = *(_DWORD *)(a1 + 128);
    v20 = v32;
    ++*(_QWORD *)(a1 + 112);
    v21 = v19 + 1;
    v22 = 2 * v18;
    v33[0] = v20;
    if ( 4 * v21 >= 3 * v18 )
    {
      v18 *= 2;
    }
    else if ( v18 - *(_DWORD *)(a1 + 132) - v21 > v18 >> 3 )
    {
LABEL_23:
      *(_DWORD *)(a1 + 128) = v21;
      if ( *v20 != -1 )
        --*(_DWORD *)(a1 + 132);
      *v20 = v31;
      result = *(unsigned int *)(a1 + 152);
      v23 = v31;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 156) )
      {
        sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), result + 1, 4u, v22, v17);
        result = *(unsigned int *)(a1 + 152);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 144) + 4 * result) = v23;
      ++*(_DWORD *)(a1 + 152);
      return result;
    }
    sub_A08C50(v2, v18);
    sub_22B31A0(v2, &v31, v33);
    v20 = (_DWORD *)v33[0];
    v21 = *(_DWORD *)(a1 + 128) + 1;
    goto LABEL_23;
  }
  if ( *(_DWORD *)(a1 + 132) )
  {
    v5 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)v5 <= 0x40 )
    {
LABEL_4:
      v6 = 4 * v5;
      v7 = *(void **)(a1 + 120);
      if ( v6 )
        memset(v7, 255, v6);
      goto LABEL_6;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 120), 4 * v5, 4);
    *(_DWORD *)(a1 + 136) = 0;
LABEL_40:
    *(_QWORD *)(a1 + 120) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 128) = 0;
  }
  *(_DWORD *)(a1 + 152) = 0;
LABEL_8:
  v8 = *(int **)(a1 + 144);
  result = (__int64)sub_2C255E0(v8, (__int64)v8, &v31);
  if ( v8 == (int *)result )
  {
    v24 = v31;
    if ( !*(_DWORD *)(a1 + 156) )
    {
      sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), 1u, 4u, v10, v11);
      v8 = (int *)(*(_QWORD *)(a1 + 144) + 4LL * *(unsigned int *)(a1 + 152));
    }
    *v8 = v24;
    result = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
    *(_DWORD *)(a1 + 152) = result;
    if ( (unsigned int)result > 2 )
    {
      v25 = *(_QWORD *)(a1 + 144) + 4 * result;
      v26 = *(int **)(a1 + 144);
      while ( 1 )
      {
        result = sub_22B31A0(v2, v26, &v32);
        if ( !(_BYTE)result )
          break;
LABEL_32:
        if ( (int *)v25 == ++v26 )
          return result;
      }
      v27 = *(_DWORD *)(a1 + 136);
      v28 = *(_DWORD *)(a1 + 128);
      v29 = v32;
      ++*(_QWORD *)(a1 + 112);
      v30 = v28 + 1;
      v33[0] = v29;
      if ( 4 * v30 >= 3 * v27 )
      {
        v27 *= 2;
      }
      else if ( v27 - *(_DWORD *)(a1 + 132) - v30 > v27 >> 3 )
      {
LABEL_36:
        *(_DWORD *)(a1 + 128) = v30;
        if ( *v29 != -1 )
          --*(_DWORD *)(a1 + 132);
        result = (unsigned int)*v26;
        *v29 = result;
        goto LABEL_32;
      }
      sub_A08C50(v2, v27);
      sub_22B31A0(v2, v26, v33);
      v29 = (_DWORD *)v33[0];
      v30 = *(_DWORD *)(a1 + 128) + 1;
      goto LABEL_36;
    }
  }
  return result;
}
