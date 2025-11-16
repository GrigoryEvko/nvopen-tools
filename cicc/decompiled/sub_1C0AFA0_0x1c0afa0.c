// Function: sub_1C0AFA0
// Address: 0x1c0afa0
//
__int64 __fastcall sub_1C0AFA0(__int64 a1, __int64 a2, int a3, char a4, __int64 a5)
{
  __int64 v5; // r15
  unsigned int v10; // esi
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 result; // rax
  __int64 v21; // rdi
  _QWORD *v22; // rdx
  unsigned int v23; // esi
  int v24; // eax
  int v25; // eax
  int v26; // r11d
  __int64 *v27; // r10
  int v28; // edi
  int v29; // edi
  int v30; // edi
  int v31; // r11d
  __int64 v32; // r10
  int v33; // edi
  __int64 v34[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1 + 8;
  v34[0] = a2;
  v10 = *(_DWORD *)(a1 + 32);
  if ( v10 )
  {
    v11 = v34[0];
    v12 = *(_QWORD *)(a1 + 16);
    v13 = (v10 - 1) & ((LODWORD(v34[0]) >> 9) ^ (LODWORD(v34[0]) >> 4));
    v14 = (__int64 *)(v12 + 24LL * v13);
    v15 = *v14;
    if ( v34[0] == *v14 )
    {
LABEL_3:
      v14[1] = a5;
      v16 = *(_DWORD *)(a1 + 32);
      if ( v16 )
        goto LABEL_4;
LABEL_22:
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_23;
    }
    v26 = 1;
    v27 = 0;
    while ( v15 != -8 )
    {
      if ( !v27 && v15 == -16 )
        v27 = v14;
      v13 = (v10 - 1) & (v26 + v13);
      v14 = (__int64 *)(v12 + 24LL * v13);
      v15 = *v14;
      if ( v34[0] == *v14 )
        goto LABEL_3;
      ++v26;
    }
    v28 = *(_DWORD *)(a1 + 24);
    if ( v27 )
      v14 = v27;
    ++*(_QWORD *)(a1 + 8);
    v29 = v28 + 1;
    if ( 4 * v29 < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 28) - v29 > v10 >> 3 )
        goto LABEL_19;
      goto LABEL_27;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 8);
  }
  v10 *= 2;
LABEL_27:
  sub_1C0AC80(v5, v10);
  sub_1C09A10(v5, v34, v35);
  v14 = (__int64 *)v35[0];
  v11 = v34[0];
  v29 = *(_DWORD *)(a1 + 24) + 1;
LABEL_19:
  *(_DWORD *)(a1 + 24) = v29;
  if ( *v14 != -8 )
    --*(_DWORD *)(a1 + 28);
  v14[1] = 0;
  *v14 = v11;
  *((_DWORD *)v14 + 4) = 0;
  v14[1] = a5;
  v16 = *(_DWORD *)(a1 + 32);
  if ( !v16 )
    goto LABEL_22;
LABEL_4:
  v17 = v34[0];
  v18 = *(_QWORD *)(a1 + 16);
  v19 = (v16 - 1) & ((LODWORD(v34[0]) >> 9) ^ (LODWORD(v34[0]) >> 4));
  result = v18 + 24LL * v19;
  v21 = *(_QWORD *)result;
  if ( *(_QWORD *)result == v34[0] )
    goto LABEL_5;
  v31 = 1;
  v32 = 0;
  while ( v21 != -8 )
  {
    if ( !v32 && v21 == -16 )
      v32 = result;
    v19 = (v16 - 1) & (v31 + v19);
    result = v18 + 24LL * v19;
    v21 = *(_QWORD *)result;
    if ( v34[0] == *(_QWORD *)result )
      goto LABEL_5;
    ++v31;
  }
  v33 = *(_DWORD *)(a1 + 24);
  if ( v32 )
    result = v32;
  ++*(_QWORD *)(a1 + 8);
  v30 = v33 + 1;
  if ( 4 * v30 < 3 * v16 )
  {
    if ( v16 - *(_DWORD *)(a1 + 28) - v30 > v16 >> 3 )
      goto LABEL_34;
    goto LABEL_24;
  }
LABEL_23:
  v16 *= 2;
LABEL_24:
  sub_1C0AC80(v5, v16);
  sub_1C09A10(v5, v34, v35);
  result = v35[0];
  v17 = v34[0];
  v30 = *(_DWORD *)(a1 + 24) + 1;
LABEL_34:
  *(_DWORD *)(a1 + 24) = v30;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 28);
  *(_QWORD *)result = v17;
  *(_QWORD *)(result + 8) = 0;
  *(_DWORD *)(result + 16) = 0;
LABEL_5:
  *(_DWORD *)(result + 16) = a3;
  if ( !a4 )
    return result;
  result = sub_1A97120(a1 + 40, v34, v35);
  v22 = (_QWORD *)v35[0];
  if ( (_BYTE)result )
    return result;
  v23 = *(_DWORD *)(a1 + 64);
  v24 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v25 = v24 + 1;
  if ( 4 * v25 >= 3 * v23 )
  {
    v23 *= 2;
    goto LABEL_38;
  }
  if ( v23 - *(_DWORD *)(a1 + 60) - v25 <= v23 >> 3 )
  {
LABEL_38:
    sub_1353F00(a1 + 40, v23);
    sub_1A97120(a1 + 40, v34, v35);
    v22 = (_QWORD *)v35[0];
    v25 = *(_DWORD *)(a1 + 56) + 1;
  }
  *(_DWORD *)(a1 + 56) = v25;
  if ( *v22 != -8 )
    --*(_DWORD *)(a1 + 60);
  result = v34[0];
  *v22 = v34[0];
  return result;
}
