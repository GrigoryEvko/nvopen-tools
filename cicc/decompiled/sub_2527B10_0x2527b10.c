// Function: sub_2527B10
// Address: 0x2527b10
//
__int64 __fastcall sub_2527B10(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, __int64 a5, _BYTE *a6)
{
  int v7; // edx
  __int64 v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int8 *v20; // rcx
  __int128 v21; // rax
  _BYTE *v22; // [rsp+8h] [rbp-88h]
  _BYTE *v23; // [rsp+8h] [rbp-88h]
  _BYTE *v24; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v25; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __m128i v28; // [rsp+30h] [rbp-60h] BYREF
  __m128i v29; // [rsp+40h] [rbp-50h] BYREF
  __int128 v30; // [rsp+50h] [rbp-40h]
  __m128i v31; // [rsp+60h] [rbp-30h] BYREF

  v28.m128i_i64[0] = a2;
  v28.m128i_i64[1] = a3;
  if ( !(_BYTE)a3 || !a2 || *(_BYTE *)a2 <= 0x15u )
    return _mm_loadu_si128(&v28).m128i_i64[0];
  if ( *(_BYTE *)a2 != 22 || *(_QWORD *)(a2 + 24) != *((_QWORD *)a4 - 4) )
    goto LABEL_6;
  v7 = *a4;
  if ( v7 == 40 )
  {
    v22 = a6;
    v26 = a4;
    v10 = sub_B491D0((__int64)a4);
    a4 = v26;
    a6 = v22;
    v9 = 32LL * v10;
  }
  else
  {
    v9 = 0;
    if ( v7 != 85 )
    {
      v9 = 64;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a4[7] & 0x80u) == 0 )
    goto LABEL_26;
  v23 = a6;
  v27 = (__int64)a4;
  v11 = sub_BD2BC0((__int64)a4);
  a4 = (unsigned __int8 *)v27;
  a6 = v23;
  v13 = v11 + v12;
  if ( *(char *)(v27 + 7) >= 0 )
  {
    if ( (unsigned int)(v13 >> 4) )
LABEL_28:
      BUG();
LABEL_26:
    v18 = 0;
    goto LABEL_20;
  }
  v14 = sub_BD2BC0(v27);
  a4 = (unsigned __int8 *)v27;
  a6 = v23;
  if ( !(unsigned int)((v13 - v14) >> 4) )
    goto LABEL_26;
  if ( *(char *)(v27 + 7) >= 0 )
    goto LABEL_28;
  v15 = *(_DWORD *)(sub_BD2BC0(v27) + 8);
  if ( *(char *)(v27 + 7) >= 0 )
    BUG();
  v16 = sub_BD2BC0(v27);
  a6 = v23;
  a4 = (unsigned __int8 *)v27;
  v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
LABEL_20:
  if ( *(_DWORD *)(a2 + 32) >= (unsigned int)((32LL * (*((_DWORD *)a4 + 1) & 0x7FFFFFF) - 32 - v9 - v18) >> 5)
    || (v24 = a6, v25 = a4, (unsigned __int8)sub_B2BB70(a2)) )
  {
LABEL_6:
    *(_QWORD *)&v30 = 0;
    BYTE8(v30) = 1;
    return v30;
  }
  v19 = *(unsigned int *)(a2 + 32);
  if ( (v25[7] & 0x40) != 0 )
    v20 = (unsigned __int8 *)*((_QWORD *)v25 - 1);
  else
    v20 = &v25[-32 * (*((_DWORD *)v25 + 1) & 0x7FFFFFF)];
  v31.m128i_i64[1] = 0;
  v31.m128i_i64[0] = (unsigned __int64)&v20[32 * v19] | 3;
  nullsub_1518();
  v29 = _mm_loadu_si128(&v31);
  *(_QWORD *)&v21 = sub_2527850(a1, &v29, a5, v24, 1u);
  return v21;
}
