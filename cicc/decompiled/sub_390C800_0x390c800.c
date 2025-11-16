// Function: sub_390C800
// Address: 0x390c800
//
__int64 __fastcall sub_390C800(__int64 a1, __int64 **a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // r15
  __int64 (*v6)(); // rax
  char v7; // r13
  __int64 v8; // r10
  __int16 v9; // dx
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rcx
  __int32 v14; // eax
  __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-B0h]
  int v19; // [rsp+Ch] [rbp-A4h]
  __int32 v20; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v21; // [rsp+24h] [rbp-8Ch] BYREF
  unsigned __int64 v22; // [rsp+28h] [rbp-88h] BYREF
  __m128i v23; // [rsp+30h] [rbp-80h] BYREF
  __int64 v24; // [rsp+40h] [rbp-70h]
  _QWORD v25[4]; // [rsp+50h] [rbp-60h] BYREF
  int v26; // [rsp+70h] [rbp-40h]
  __int64 v27; // [rsp+78h] [rbp-38h]

  v3 = a1;
  v5 = **a2;
  v19 = *(_DWORD *)(a3 + 72);
  v6 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 72LL);
  if ( v6 == sub_38CB1C0 || !(unsigned __int8)v6() )
    v7 = sub_38CF260(*(_QWORD *)(a3 + 136), &v22, a2);
  else
    v7 = sub_38CF2A0(*(_QWORD *)(a3 + 136), &v22, a2);
  *(_DWORD *)(a3 + 72) = 0;
  v8 = *(_QWORD *)(a3 + 128);
  v26 = 1;
  v25[0] = &unk_49EFC48;
  v18 = v8;
  v27 = a3 + 64;
  memset(&v25[1], 0, 24);
  sub_16E7A40((__int64)v25, 0, 0, 0);
  *(_DWORD *)(a3 + 96) = 0;
  v9 = *(_WORD *)(a1 + 176);
  if ( v7 )
  {
    v23.m128i_i8[2] = *(_BYTE *)(a1 + 178);
    v23.m128i_i16[0] = v9;
    sub_38C6700(v5, v23.m128i_i32[0], v18, v22, (__int64)v25);
  }
  else
  {
    if ( (unsigned __int8)sub_38C95E0(
                            v5,
                            ((unsigned __int64)HIBYTE(v9) << 8)
                          | (unsigned __int8)v9
                          | ((unsigned __int64)*(unsigned __int8 *)(a1 + 178) << 16),
                            v18,
                            v22,
                            (__int64)v25,
                            &v20,
                            &v21) )
      v13 = *(_QWORD *)(a3 + 136);
    else
      v13 = *(_QWORD *)(*(_QWORD *)(a3 + 136) + 24LL);
    v14 = 2;
    if ( v21 != 4 )
    {
      v14 = 3;
      if ( v21 <= 4 )
        v14 = v21 != 1;
    }
    v23.m128i_i32[3] = v14;
    v23.m128i_i64[0] = v13;
    v15 = *(unsigned int *)(a3 + 96);
    v23.m128i_i32[2] = v20;
    v24 = 0;
    if ( (unsigned int)v15 >= *(_DWORD *)(a3 + 100) )
    {
      sub_16CD150(a3 + 88, (const void *)(a3 + 104), 0, 24, v11, v12);
      v15 = *(unsigned int *)(a3 + 96);
    }
    v16 = (__m128i *)(*(_QWORD *)(a3 + 88) + 24 * v15);
    v17 = v24;
    *v16 = _mm_loadu_si128(&v23);
    v16[1].m128i_i64[0] = v17;
    ++*(_DWORD *)(a3 + 96);
  }
  LOBYTE(v3) = *(_DWORD *)(a3 + 72) != v19;
  v25[0] = &unk_49EFD28;
  sub_16E7960((__int64)v25);
  return v3;
}
