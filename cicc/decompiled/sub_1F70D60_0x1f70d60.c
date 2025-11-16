// Function: sub_1F70D60
// Address: 0x1f70d60
//
__int64 __fastcall sub_1F70D60(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __m128i *a5)
{
  unsigned int v8; // r13d
  __int64 v9; // r12
  __int64 v12; // rdx
  char v13; // di
  __int64 v14; // rax
  unsigned int v15; // eax
  char v16; // dl
  unsigned int v17; // r12d
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  unsigned int v27; // eax
  char v28; // [rsp+0h] [rbp-60h]
  unsigned __int8 v29; // [rsp+0h] [rbp-60h]
  char v30; // [rsp+0h] [rbp-60h]
  char v32[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v33; // [rsp+18h] [rbp-48h]
  __m128i v34[4]; // [rsp+20h] [rbp-40h] BYREF

  v8 = *(_DWORD *)(a2 + 32);
  if ( v8 > 0x40 )
  {
    LODWORD(_R12) = sub_16A58F0(a2 + 24);
    if ( !(_DWORD)_R12 || v8 != (_DWORD)_R12 + (unsigned int)sub_16A57B0(a2 + 24) )
      return 0;
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 24);
    if ( !v9 || (v9 & (v9 + 1)) != 0 )
      return 0;
    if ( !~v9 )
      goto LABEL_41;
    __asm { tzcnt   r12, r12 }
  }
  if ( (_DWORD)_R12 == 32 )
  {
    v12 = 5;
    goto LABEL_10;
  }
  if ( (unsigned int)_R12 > 0x20 )
  {
    if ( (_DWORD)_R12 != 64 )
    {
      if ( (_DWORD)_R12 != 128 )
        goto LABEL_28;
      v12 = 7;
      goto LABEL_10;
    }
LABEL_41:
    v12 = 6;
    goto LABEL_10;
  }
  if ( (_DWORD)_R12 == 8 )
  {
    v12 = 3;
    goto LABEL_10;
  }
  v12 = 4;
  if ( (_DWORD)_R12 == 16 || (v12 = 2, (_DWORD)_R12 == 1) )
  {
LABEL_10:
    a5->m128i_i8[0] = v12;
    a5->m128i_i64[1] = 0;
    v13 = *(_BYTE *)(a3 + 88);
    v14 = *(_QWORD *)(a3 + 96);
    v32[0] = v13;
    v33 = v14;
    if ( v13 == (_BYTE)v12 )
      return !*(_BYTE *)(a1 + 24)
          || a4 && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 2 * (v12 + 115LL * a4 + 16104) + 1) & 0xF0) == 0;
    goto LABEL_11;
  }
LABEL_28:
  v23 = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)a1 + 48LL), _R12);
  v25 = v24;
  a5->m128i_i8[0] = v23;
  v12 = v23;
  a5->m128i_i64[1] = v25;
  v13 = *(_BYTE *)(a3 + 88);
  v26 = *(_QWORD *)(a3 + 96);
  v32[0] = v13;
  v33 = v26;
  if ( v13 == v23 )
  {
    if ( !v13 )
    {
      if ( v26 == v25 )
        return !*(_BYTE *)(a1 + 24);
      LOBYTE(v12) = *(_BYTE *)(a3 + 26) & 8;
      if ( (_BYTE)v12 )
        return 0;
      v34[0] = _mm_loadu_si128(a5);
      goto LABEL_33;
    }
    return !*(_BYTE *)(a1 + 24)
        || a4 && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 2 * (v12 + 115LL * a4 + 16104) + 1) & 0xF0) == 0;
  }
LABEL_11:
  if ( (*(_BYTE *)(a3 + 26) & 8) != 0 )
    return 0;
  v34[0] = _mm_loadu_si128(a5);
  if ( v13 )
  {
    v28 = v12;
    v15 = sub_1F6C8D0(v13);
    v16 = v28;
    v17 = v15;
    goto LABEL_14;
  }
LABEL_33:
  v30 = v12;
  v27 = sub_1F58D40((__int64)v32);
  v16 = v30;
  v17 = v27;
LABEL_14:
  if ( !v16 )
  {
    if ( (unsigned int)sub_1F58D40((__int64)v34) < v17 )
    {
      v18 = sub_1F58D40((__int64)a5);
      v19 = 0;
      goto LABEL_16;
    }
    return 0;
  }
  v29 = v16;
  v18 = sub_1F6C8D0(v16);
  v19 = v29;
  if ( v18 >= v17 )
    return 0;
LABEL_16:
  if ( v18 <= 7 )
    return 0;
  if ( (v18 & (v18 - 1)) != 0 )
    return 0;
  v20 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( !(_BYTE)v19 || !a4 || (*(_BYTE *)(v20 + 2 * (v19 + 115LL * a4 + 16104) + 1) & 0xF0) != 0 )
      return 0;
  }
  v21 = *(__int64 (**)())(*(_QWORD *)v20 + 416LL);
  if ( v21 == sub_1F3CAB0 )
    return 1;
  return ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64))v21)(
           v20,
           a3,
           3,
           a5->m128i_u32[0],
           a5->m128i_i64[1]);
}
