// Function: sub_3287910
// Address: 0x3287910
//
unsigned __int64 __fastcall sub_3287910(__int64 a1, unsigned int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 *v8; // rsi
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rcx
  unsigned int v15; // r9d
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v19; // rcx
  unsigned int v21; // r13d
  bool v24; // zf
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rbx
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // [rsp-58h] [rbp-58h]
  __int64 v32; // [rsp-58h] [rbp-58h]
  unsigned int v33; // [rsp-50h] [rbp-50h]
  unsigned int v34; // [rsp-50h] [rbp-50h]
  __int64 v35; // [rsp-48h] [rbp-48h] BYREF
  __int64 v36; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 24) != 186 )
    return 0;
  v8 = *(__int64 **)(a1 + 40);
  v9 = v8[5];
  v10 = *(_DWORD *)(v9 + 24);
  if ( v10 != 11 && v10 != 35 )
    return 0;
  v11 = *v8;
  if ( *(_DWORD *)(*v8 + 24) != 298 )
    return 0;
  if ( (*(_BYTE *)(v11 + 33) & 0xC) != 0 )
    return 0;
  if ( (*(_WORD *)(v11 + 32) & 0x380) != 0 )
    return 0;
  v12 = *(_QWORD *)(v11 + 40);
  if ( *(_QWORD *)(v12 + 40) != a3 )
    return 0;
  if ( *(_DWORD *)(v12 + 48) != a4 )
    return 0;
  v13 = *(_WORD *)(*(_QWORD *)(a1 + 48) + 16LL * a2);
  if ( (unsigned __int16)(v13 - 6) > 2u )
    return 0;
  v14 = *(_QWORD *)(v9 + 96);
  v15 = *(_DWORD *)(v14 + 32);
  v16 = *(__int64 **)(v14 + 24);
  if ( v15 > 0x40 )
  {
    v17 = *v16;
  }
  else
  {
    if ( !v15 )
    {
      v21 = 0;
      v25 = 64;
      goto LABEL_20;
    }
    v17 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
  }
  _RAX = ~v17;
  if ( !_RAX )
    return 0;
  _BitScanReverse64(&v19, _RAX);
  v15 = v19 ^ 0x3F;
  if ( (((unsigned int)v19 ^ 0x3F) & 7) != 0 )
    return 0;
  __asm { tzcnt   rcx, rax }
  v21 = _RCX;
  if ( (_RCX & 7) != 0 )
    return 0;
  _RAX = ~(_RAX >> _RCX);
  __asm { tzcnt   rcx, rax }
  v24 = _RAX == 0;
  v25 = 64;
  if ( !v24 )
    v25 = _RCX;
LABEL_20:
  if ( v15 + v21 + v25 != 64 )
    return 0;
  if ( v13 != 8 && v15 )
  {
    v32 = a5;
    v34 = v15;
    v35 = sub_3262090(a1, a2);
    v36 = v29;
    v30 = sub_CA1930(&v35);
    a5 = v32;
    v15 = v34 + v30 - 64;
  }
  v31 = a5;
  v33 = v15;
  v26 = sub_3262090(a1, a2);
  v36 = v27;
  v35 = v26;
  v28 = (sub_CA1930(&v35) - v21 - (unsigned __int64)v33) >> 3;
  if ( (unsigned int)v28 <= 2 )
  {
    if ( (_DWORD)v28 )
      goto LABEL_26;
    return 0;
  }
  if ( (_DWORD)v28 != 4 )
    return 0;
LABEL_26:
  if ( v21 && (v21 >> 3) % (unsigned int)v28 )
    return 0;
  if ( v11 != v31 )
  {
    if ( *(_DWORD *)(v31 + 24) != 2 )
      return 0;
    v35 = v11;
    LODWORD(v36) = 1;
    if ( !(unsigned __int8)sub_3286E00(&v35) || !(unsigned __int8)sub_33CFA90(v11, v31) )
      return 0;
  }
  return ((unsigned __int64)(v21 >> 3) << 32) | (unsigned int)v28;
}
