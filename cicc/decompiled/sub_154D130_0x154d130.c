// Function: sub_154D130
// Address: 0x154d130
//
void __fastcall sub_154D130(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rbx
  unsigned int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rcx
  char v16; // dl
  int v17; // r10d
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdi
  __int64 v25; // r11
  __int64 v26; // [rsp+0h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+8h] [rbp-38h] BYREF

  v10 = a1;
  v11 = *(_DWORD *)(a3 + 24);
  v26 = a1;
  if ( !v11 )
  {
    ++*(_QWORD *)a3;
LABEL_29:
    v11 *= 2;
LABEL_30:
    sub_1541430(a3, v11);
    sub_154C3E0(a3, &v26, v27);
    v14 = v27[0];
    v20 = v26;
    v19 = *(_DWORD *)(a3 + 16) + 1;
    goto LABEL_12;
  }
  v12 = *(_QWORD *)(a3 + 8);
  v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v14 = v12 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( v10 == *(_QWORD *)v14 )
  {
    v16 = *(_BYTE *)(v14 + 12);
LABEL_4:
    if ( v16 )
      return;
    goto LABEL_15;
  }
  v17 = 1;
  a7 = 0;
  while ( v15 != -8 )
  {
    if ( v15 != -16 || a7 )
      v14 = a7;
    a7 = (unsigned int)(v17 + 1);
    v13 = (v11 - 1) & (v17 + v13);
    v25 = v12 + 16LL * v13;
    v15 = *(_QWORD *)v25;
    if ( v10 == *(_QWORD *)v25 )
    {
      v16 = *(_BYTE *)(v25 + 12);
      v14 = v25;
      goto LABEL_4;
    }
    ++v17;
    a7 = v14;
    v14 = v12 + 16LL * v13;
  }
  v18 = *(_DWORD *)(a3 + 16);
  if ( a7 )
    v14 = a7;
  ++*(_QWORD *)a3;
  v19 = v18 + 1;
  if ( 4 * v19 >= 3 * v11 )
    goto LABEL_29;
  v20 = v10;
  if ( v11 - *(_DWORD *)(a3 + 20) - v19 <= v11 >> 3 )
    goto LABEL_30;
LABEL_12:
  *(_DWORD *)(a3 + 16) = v19;
  if ( *(_QWORD *)v14 != -8 )
    --*(_DWORD *)(a3 + 20);
  *(_QWORD *)v14 = v20;
  *(_DWORD *)(v14 + 8) = 0;
  *(_BYTE *)(v14 + 12) = 0;
LABEL_15:
  *(_BYTE *)(v14 + 12) = 1;
  v21 = *(_QWORD *)(v10 + 8);
  if ( v21 && *(_QWORD *)(v21 + 8) )
  {
    sub_154C490(v10, a2, *(_DWORD *)(v14 + 8), a3, a4, a7, a5);
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
      return;
  }
  else if ( *(_BYTE *)(v10 + 16) > 0x10u )
  {
    return;
  }
  if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 0 )
  {
    v22 = 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    {
      v23 = *(_QWORD *)(v10 - 8);
      v10 = v23 + v22;
    }
    else
    {
      v23 = v10 - v22;
    }
    v24 = *(_QWORD *)v23;
    if ( *(_BYTE *)(*(_QWORD *)v23 + 16LL) <= 0x10u )
      goto LABEL_23;
    while ( 1 )
    {
      v23 += 24;
      if ( v10 == v23 )
        break;
      v24 = *(_QWORD *)v23;
      if ( *(_BYTE *)(*(_QWORD *)v23 + 16LL) <= 0x10u )
LABEL_23:
        sub_154D130(v24, a2, a3, a4);
    }
  }
}
