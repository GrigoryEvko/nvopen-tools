// Function: sub_19110A0
// Address: 0x19110a0
//
__int64 __fastcall sub_19110A0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 result; // rax
  __int64 v9; // rcx
  unsigned int v10; // esi
  int v11; // ecx
  __int64 v12; // r8
  unsigned int v13; // edx
  int v14; // edi
  int v15; // r10d
  __int64 v16; // r9
  int v17; // eax
  int v18; // edx
  __int64 v19; // rcx
  int v20; // r11d
  __int64 v21; // r10
  int v22; // edi
  int v23; // edi
  int v24[3]; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h] BYREF
  __int64 v26; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-28h]

  v26 = a2;
  v5 = *(_DWORD *)(a1 + 24);
  v24[0] = a3;
  v27 = a3;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_17:
    v5 *= 2;
    goto LABEL_18;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v6 + 16LL * v7;
  v9 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    goto LABEL_3;
  v15 = 1;
  v16 = 0;
  while ( v9 != -8 )
  {
    if ( v16 || v9 != -16 )
      result = v16;
    v7 = (v5 - 1) & (v15 + v7);
    v9 = *(_QWORD *)(v6 + 16LL * v7);
    if ( a2 == v9 )
      goto LABEL_3;
    ++v15;
    v16 = result;
    result = v6 + 16LL * v7;
  }
  if ( !v16 )
    v16 = result;
  v17 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v5 )
    goto LABEL_17;
  v19 = a2;
  if ( v5 - *(_DWORD *)(a1 + 20) - v18 <= v5 >> 3 )
  {
LABEL_18:
    sub_177C7D0(a1, v5);
    sub_190E590(a1, &v26, &v25);
    v16 = v25;
    v19 = v26;
    v18 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v18;
  if ( *(_QWORD *)v16 != -8 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v16 = v19;
  result = v27;
  *(_DWORD *)(v16 + 8) = v27;
LABEL_3:
  if ( *(_BYTE *)(a2 + 16) != 77 )
    return result;
  v10 = *(_DWORD *)(a1 + 144);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_29;
  }
  v11 = v24[0];
  v12 = *(_QWORD *)(a1 + 128);
  v13 = (v10 - 1) & (37 * v24[0]);
  result = v12 + 16LL * v13;
  v14 = *(_DWORD *)result;
  if ( *(_DWORD *)result != v24[0] )
  {
    v20 = 1;
    v21 = 0;
    while ( v14 != -1 )
    {
      if ( !v21 && v14 == -2 )
        v21 = result;
      v13 = (v10 - 1) & (v20 + v13);
      result = v12 + 16LL * v13;
      v14 = *(_DWORD *)result;
      if ( v24[0] == *(_DWORD *)result )
        goto LABEL_6;
      ++v20;
    }
    v22 = *(_DWORD *)(a1 + 136);
    if ( v21 )
      result = v21;
    ++*(_QWORD *)(a1 + 120);
    v23 = v22 + 1;
    if ( 4 * v23 < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 140) - v23 > v10 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 136) = v23;
        if ( *(_DWORD *)result != -1 )
          --*(_DWORD *)(a1 + 140);
        *(_DWORD *)result = v11;
        *(_QWORD *)(result + 8) = 0;
        goto LABEL_6;
      }
LABEL_30:
      sub_1910EE0(a1 + 120, v10);
      sub_190E640(a1 + 120, v24, &v26);
      result = v26;
      v11 = v24[0];
      v23 = *(_DWORD *)(a1 + 136) + 1;
      goto LABEL_25;
    }
LABEL_29:
    v10 *= 2;
    goto LABEL_30;
  }
LABEL_6:
  *(_QWORD *)(result + 8) = a2;
  return result;
}
