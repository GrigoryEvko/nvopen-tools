// Function: sub_3742340
// Address: 0x3742340
//
__int64 __fastcall sub_3742340(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int8 **a5)
{
  __int32 v10; // eax
  unsigned int v11; // r10d
  __int64 v12; // rdi
  unsigned __int8 v13; // al
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rcx
  int v18; // esi
  unsigned int v19; // eax
  __int64 v20; // rdx
  int v21; // r8d
  unsigned __int8 v22; // [rsp+Fh] [rbp-91h]
  unsigned __int64 v23[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v24[4]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v25; // [rsp+40h] [rbp-60h] BYREF
  __int128 v26; // [rsp+50h] [rbp-50h]
  __int128 v27; // [rsp+60h] [rbp-40h]

  v25 = 0;
  v26 = 0;
  v27 = 0;
  v10 = sub_3742170(a1, a2);
  if ( v10 )
  {
LABEL_7:
    v25.m128i_i64[0] &= 0xFFFFFFF000000000LL;
    v25.m128i_i32[2] = v10;
    v26 = 0u;
    *(_QWORD *)&v27 = 0;
    if ( !BYTE8(v27) )
      BYTE8(v27) = 1;
    goto LABEL_9;
  }
  v11 = BYTE8(v27);
  if ( !BYTE8(v27) )
  {
    if ( !*(_QWORD *)(a2 + 16) || *(_BYTE *)a2 <= 0x1Cu )
      return v11;
    v12 = *(_QWORD *)(a1 + 40);
    if ( *(_BYTE *)a2 == 60 )
    {
      v16 = *(_DWORD *)(v12 + 272);
      v17 = *(_QWORD *)(v12 + 256);
      if ( v16 )
      {
        v18 = v16 - 1;
        v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = *(_QWORD *)(v17 + 16LL * v19);
        if ( a2 == v20 )
          return v11;
        v21 = 1;
        while ( v20 != -4096 )
        {
          v19 = v18 & (v21 + v19);
          v20 = *(_QWORD *)(v17 + 16LL * v19);
          if ( a2 == v20 )
            return v11;
          ++v21;
        }
      }
    }
    v10 = sub_374D810(v12, a2);
    goto LABEL_7;
  }
LABEL_9:
  v13 = sub_2E799E0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL));
  if ( v13 && !v25.m128i_i8[0] )
  {
    v22 = v13;
    v23[0] = (unsigned __int64)v24;
    v23[1] = 0x300000003LL;
    v24[0] = 4101;
    v24[1] = 0;
    v24[2] = 6;
    v15 = sub_B0D8A0(a3, (__int64)v23, 0, 0);
    sub_2E90D80(
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL),
      *(unsigned __int64 **)(*(_QWORD *)(a1 + 40) + 752LL),
      a5,
      (_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL) - 640LL),
      0,
      a4,
      &v25,
      1,
      v15);
    v11 = v22;
    if ( (_QWORD *)v23[0] != v24 )
    {
      _libc_free(v23[0]);
      return v22;
    }
  }
  else
  {
    sub_2E90D80(
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL),
      *(unsigned __int64 **)(*(_QWORD *)(a1 + 40) + 752LL),
      a5,
      (_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL) - 560LL),
      1,
      a4,
      &v25,
      1,
      (__int64)a3);
    return 1;
  }
  return v11;
}
