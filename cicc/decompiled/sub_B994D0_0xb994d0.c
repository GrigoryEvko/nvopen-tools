// Function: sub_B994D0
// Address: 0xb994d0
//
void __fastcall sub_B994D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r13d
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int v11; // eax
  __int64 *v12; // rdi
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v19; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  v5 = a1;
  v6 = *(_BYTE *)(a1 + 7);
  if ( (v6 & 0x20) == 0 )
    *(_BYTE *)(a1 + 7) = v6 | 0x20;
  v7 = *(_QWORD *)sub_BD5C60(a1, a2);
  v18 = a1;
  v8 = *(_DWORD *)(v7 + 3248);
  if ( !v8 )
  {
    ++*(_QWORD *)(v7 + 3224);
    v19 = 0;
LABEL_21:
    v8 *= 2;
    goto LABEL_22;
  }
  v9 = *(_QWORD *)(v7 + 3232);
  v10 = 1;
  v11 = (v8 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = (__int64 *)(v9 + 40LL * v11);
  v13 = 0;
  v14 = *v12;
  if ( v5 == *v12 )
  {
LABEL_5:
    v15 = (__int64)(v12 + 1);
    goto LABEL_6;
  }
  while ( v14 != -4096 )
  {
    if ( !v13 && v14 == -8192 )
      v13 = v12;
    v11 = (v8 - 1) & (v10 + v11);
    v12 = (__int64 *)(v9 + 40LL * v11);
    v14 = *v12;
    if ( v5 == *v12 )
      goto LABEL_5;
    ++v10;
  }
  v16 = *(_DWORD *)(v7 + 3240);
  if ( !v13 )
    v13 = v12;
  ++*(_QWORD *)(v7 + 3224);
  v17 = v16 + 1;
  v19 = v13;
  if ( 4 * (v16 + 1) >= 3 * v8 )
    goto LABEL_21;
  if ( v8 - *(_DWORD *)(v7 + 3244) - v17 <= v8 >> 3 )
  {
LABEL_22:
    sub_B98D30(v7 + 3224, v8);
    sub_B92880(v7 + 3224, &v18, &v19);
    v5 = v18;
    v13 = v19;
    v17 = *(_DWORD *)(v7 + 3240) + 1;
  }
  *(_DWORD *)(v7 + 3240) = v17;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v7 + 3244);
  *v13 = v5;
  v15 = (__int64)(v13 + 1);
  v13[1] = (__int64)(v13 + 3);
  v13[2] = 0x100000000LL;
LABEL_6:
  sub_B97C00(v15, v4, a3);
}
