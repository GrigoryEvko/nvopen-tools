// Function: sub_159B790
// Address: 0x159b790
//
__int64 __fastcall sub_159B790(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r9
  __int64 v10; // rdx
  int v11; // r11d
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned int i; // eax
  __int64 *v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 result; // rax
  __int64 v19; // r12
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // edx
  int v28; // edx
  int v29; // r11d
  unsigned int v30; // r10d
  unsigned __int64 v31; // r9
  unsigned __int64 v32; // r9
  unsigned int j; // eax
  _QWORD *v34; // r9
  unsigned int v35; // eax
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 *v44; // [rsp+18h] [rbp-48h] BYREF
  __int64 v45; // [rsp+20h] [rbp-40h] BYREF
  __int64 v46; // [rsp+28h] [rbp-38h]

  v3 = a3;
  v5 = *(_QWORD *)(a1 - 48);
  if ( v5 == a2 )
  {
    v19 = *(_QWORD *)(a1 - 24);
    v5 = sub_1649C60(a3);
    v3 = v19;
  }
  v45 = v5;
  v46 = v3;
  v6 = sub_16498A0(a1);
  v7 = *(_QWORD *)v6;
  v8 = *(_DWORD *)(*(_QWORD *)v6 + 1768LL);
  v9 = *(_QWORD *)v6 + 1744LL;
  if ( !v8 )
  {
    ++*(_QWORD *)(v7 + 1744);
LABEL_51:
    v8 *= 2;
LABEL_52:
    v43 = v9;
    sub_159B4E0(v9, v8);
    sub_15977E0(v43, &v45, &v44);
    v15 = v44;
    v10 = v45;
    v21 = *(_DWORD *)(v7 + 1760) + 1;
    goto LABEL_21;
  }
  v10 = v45;
  v42 = 0;
  v11 = 1;
  v12 = (((((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)
         | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)
        | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)) << 32));
  v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
      ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
  for ( i = (v8 - 1) & (((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - ((_DWORD)v13 << 27))); ; i = (v8 - 1) & v17 )
  {
    v15 = (__int64 *)(*(_QWORD *)(v7 + 1752) + 24LL * i);
    v16 = *v15;
    if ( *v15 == v45 && v15[1] == v46 )
    {
      result = v15[2];
      if ( !result )
        goto LABEL_24;
      return result;
    }
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && v15[1] == -16 )
    {
      if ( v42 )
        v15 = v42;
      v42 = v15;
    }
LABEL_12:
    v17 = v11 + i;
    ++v11;
  }
  if ( v15[1] != -8 )
    goto LABEL_12;
  if ( v42 )
    v15 = v42;
  v20 = *(_DWORD *)(v7 + 1760);
  ++*(_QWORD *)(v7 + 1744);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v8 )
    goto LABEL_51;
  if ( v8 - *(_DWORD *)(v7 + 1764) - v21 <= v8 >> 3 )
    goto LABEL_52;
LABEL_21:
  *(_DWORD *)(v7 + 1760) = v21;
  if ( *v15 != -8 || v15[1] != -8 )
    --*(_DWORD *)(v7 + 1764);
  *v15 = v10;
  v22 = v46;
  v15[2] = 0;
  v15[1] = v22;
LABEL_24:
  --*(_WORD *)(*(_QWORD *)(a1 - 24) + 18LL);
  v23 = sub_16498A0(a1);
  v24 = *(_QWORD *)(a1 - 24);
  v25 = *(_QWORD *)(a1 - 48);
  v26 = *(_QWORD *)v23;
  v27 = *(_DWORD *)(*(_QWORD *)v23 + 1768LL);
  if ( v27 )
  {
    v28 = v27 - 1;
    v29 = 1;
    v30 = (unsigned int)v24 >> 9;
    v31 = (((v30 ^ ((unsigned int)v24 >> 4)
           | ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v30 ^ ((unsigned int)v24 >> 4)) << 32)) >> 22)
        ^ ((v30 ^ ((unsigned int)v24 >> 4)
          | ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v30 ^ ((unsigned int)v24 >> 4)) << 32));
    v32 = ((9 * (((v31 - 1 - (v31 << 13)) >> 8) ^ (v31 - 1 - (v31 << 13)))) >> 15)
        ^ (9 * (((v31 - 1 - (v31 << 13)) >> 8) ^ (v31 - 1 - (v31 << 13))));
    for ( j = v28 & (((v32 - 1 - (v32 << 27)) >> 31) ^ (v32 - 1 - ((_DWORD)v32 << 27))); ; j = v28 & v35 )
    {
      v34 = (_QWORD *)(*(_QWORD *)(v26 + 1752) + 24LL * j);
      if ( v25 == *v34 && v24 == v34[1] )
        break;
      if ( *v34 == -8 && v34[1] == -8 )
        goto LABEL_31;
      v35 = v29 + j;
      ++v29;
    }
    *v34 = -16;
    v34[1] = -16;
    --*(_DWORD *)(v26 + 1760);
    ++*(_DWORD *)(v26 + 1764);
  }
LABEL_31:
  v15[2] = a1;
  if ( *(_QWORD *)(a1 - 48) )
  {
    v36 = *(_QWORD *)(a1 - 40);
    v37 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v37 = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
  }
  *(_QWORD *)(a1 - 48) = v5;
  if ( v5 )
  {
    v38 = *(_QWORD *)(v5 + 8);
    *(_QWORD *)(a1 - 40) = v38;
    if ( v38 )
      *(_QWORD *)(v38 + 16) = (a1 - 40) | *(_QWORD *)(v38 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (v5 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(v5 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v39 = *(_QWORD *)(a1 - 16);
    v40 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v40 = v39;
    if ( v39 )
      *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
  }
  *(_QWORD *)(a1 - 24) = v3;
  if ( v3 )
  {
    v41 = *(_QWORD *)(v3 + 8);
    *(_QWORD *)(a1 - 16) = v41;
    if ( v41 )
      *(_QWORD *)(v41 + 16) = (a1 - 16) | *(_QWORD *)(v41 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (v3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(v3 + 8) = a1 - 24;
    v3 = *(_QWORD *)(a1 - 24);
  }
  ++*(_WORD *)(v3 + 18);
  return 0;
}
