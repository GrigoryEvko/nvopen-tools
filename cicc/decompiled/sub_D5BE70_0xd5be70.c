// Function: sub_D5BE70
// Address: 0xd5be70
//
__m128i *__fastcall sub_D5BE70(__m128i *a1, unsigned __int8 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int32 v7; // edx
  int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __m128i v17; // xmm0
  __int32 v18; // eax
  __int32 v20; // ecx
  __int64 v21; // rax
  __int32 v22; // edx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-48h] BYREF
  __m128i v25; // [rsp+10h] [rbp-40h] BYREF
  __int64 v26; // [rsp+20h] [rbp-30h]
  __int32 v27; // [rsp+28h] [rbp-28h]

  v4 = sub_D5BAA0(a2);
  if ( v4 )
  {
    if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v4 + 24) + 16LL) + 8LL) == 14 )
    {
      sub_D5BC90(&v25, v4, 7u, a3);
      if ( (_BYTE)v27 )
      {
        v17 = _mm_loadu_si128(&v25);
        a1[1].m128i_i64[0] = v26;
        v18 = v27;
        *a1 = v17;
        a1[1].m128i_i32[2] = v18;
        return a1;
      }
    }
  }
  v25.m128i_i64[0] = *((_QWORD *)a2 + 9);
  v5 = sub_A747F0(&v25, -1, 88);
  if ( v5 )
  {
    v24 = v5;
  }
  else
  {
    v24 = sub_B495C0((__int64)a2, 88);
    if ( !v24 )
    {
      a1[1].m128i_i8[8] = 0;
      return a1;
    }
  }
  v6 = sub_A71E50(&v24);
  v25.m128i_i32[2] = v7;
  v8 = *a2;
  v25.m128i_i64[0] = v6;
  if ( v8 == 40 )
  {
    v9 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = 0;
    if ( v8 != 85 )
    {
      v9 = 64;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v10 = sub_BD2BC0((__int64)a2);
    v12 = v10 + v11;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v12 >> 4) )
        goto LABEL_27;
    }
    else if ( (unsigned int)((v12 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v14 = sub_BD2BC0((__int64)a2);
        v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
        goto LABEL_19;
      }
LABEL_27:
      BUG();
    }
  }
  v16 = 0;
LABEL_19:
  v20 = v25.m128i_i32[0];
  v21 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v9 - v16;
  v22 = -1;
  v23 = v21 >> 5;
  if ( v25.m128i_i8[8] )
    v22 = v25.m128i_i32[1];
  a1->m128i_i32[1] = v23;
  a1->m128i_i8[0] = 2;
  a1->m128i_i32[2] = v20;
  a1->m128i_i32[3] = v22;
  a1[1].m128i_i32[0] = -1;
  a1[1].m128i_i32[1] = 0;
  a1[1].m128i_i8[8] = 1;
  return a1;
}
