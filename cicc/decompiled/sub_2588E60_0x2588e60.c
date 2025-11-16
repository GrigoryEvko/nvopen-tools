// Function: sub_2588E60
// Address: 0x2588e60
//
__int64 __fastcall sub_2588E60(__int64 a1, char *a2, _BYTE *a3)
{
  unsigned __int8 *v4; // rdi
  int v5; // eax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 *v14; // rdx
  __int64 v15; // r8
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __m128i v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  char v21; // bl
  char v22; // al
  __int64 v23; // rdx
  _BYTE *v24; // rax
  int v25; // edx
  bool v26; // [rsp+15h] [rbp-3Bh] BYREF
  char v27; // [rsp+16h] [rbp-3Ah] BYREF
  unsigned __int8 *v28; // [rsp+18h] [rbp-38h] BYREF
  __m128i v29[3]; // [rsp+20h] [rbp-30h] BYREF

  v4 = (unsigned __int8 *)*((_QWORD *)a2 + 3);
  v5 = *v4;
  if ( (_BYTE)v5 == 61 )
    return 1;
  if ( (_BYTE)v5 == 62 )
  {
    if ( *((_QWORD *)v4 - 8) != *(_QWORD *)a2 )
      return 1;
    goto LABEL_22;
  }
  if ( (unsigned __int8)(v5 - 34) > 0x33u || (v8 = 0x8000000000041LL, !_bittest64(&v8, (unsigned int)(v5 - 34))) )
  {
    v16 = (unsigned int)(v5 - 63);
    if ( (unsigned __int8)v16 <= 0x17u )
    {
      v17 = 10518529;
      if ( _bittest64(&v17, v16) )
      {
        *a3 = 1;
        return 1;
      }
    }
    goto LABEL_22;
  }
  v28 = v4;
  if ( !sub_254C190(v4, (unsigned __int64)a2) || sub_B46A10((__int64)v28) )
    return 1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(v10 + 160);
  v12 = *(unsigned int *)(v10 + 176);
  if ( (_DWORD)v12 )
  {
    v13 = ((_DWORD)v12 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v14 = (__int64 *)(v11 + 16 * v13);
    v15 = *v14;
    if ( v28 == (unsigned __int8 *)*v14 )
    {
LABEL_11:
      if ( v14 != (__int64 *)(v11 + 16 * v12) )
      {
        sub_2575530(*(_QWORD *)(a1 + 16) + 24LL, (__int64 *)&v28, (__int64)v14, v13, v15, v9);
        return 1;
      }
    }
    else
    {
      v25 = 1;
      while ( v15 != -4096 )
      {
        v9 = (unsigned int)(v25 + 1);
        v13 = ((_DWORD)v12 - 1) & (unsigned int)(v25 + v13);
        v14 = (__int64 *)(v11 + 16LL * (unsigned int)v13);
        v15 = *v14;
        if ( v28 == (unsigned __int8 *)*v14 )
          goto LABEL_11;
        v25 = v9;
      }
    }
  }
  v18.m128i_i64[0] = sub_254C9B0((__int64)v28, (a2 - (char *)&v28[-32 * (*((_DWORD *)v28 + 1) & 0x7FFFFFF)]) >> 5);
  v19 = *(_QWORD *)(a1 + 8);
  v20 = *(_QWORD *)(a1 + 24);
  v29[0] = v18;
  v21 = sub_2588040(v20, v19, v29[0].m128i_i64, 1, &v26, 0, 0);
  v22 = sub_257BF90(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v29, 1, &v27, 0, 0);
  v23 = *(_QWORD *)(a1 + 16);
  if ( v21 && (*(_DWORD *)(v23 + 8) == 109 || v22) )
    return 1;
  *(_BYTE *)(v23 + 16) |= v22 ^ 1;
  v24 = *(_BYTE **)a1;
  if ( **(_BYTE **)a1 && *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) == 109 )
  {
    sub_2560480(*(_QWORD **)(a1 + 24), (__int64)v28, "OMP113", 6u);
LABEL_22:
    v24 = *(_BYTE **)a1;
  }
  *v24 = 0;
  return 1;
}
