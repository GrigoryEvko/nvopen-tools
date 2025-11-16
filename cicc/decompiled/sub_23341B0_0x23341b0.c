// Function: sub_23341B0
// Address: 0x23341b0
//
__int64 __fastcall sub_23341B0(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i v4; // xmm1
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdi
  char v11; // r8
  char v12; // al
  char v13; // cl
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  char v18; // [rsp+6h] [rbp-CAh]
  char v19; // [rsp+7h] [rbp-C9h]
  __int32 v20; // [rsp+8h] [rbp-C8h]
  unsigned int v21; // [rsp+8h] [rbp-C8h]
  unsigned int v22; // [rsp+8h] [rbp-C8h]
  __m128i v23; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+28h] [rbp-A8h] BYREF
  __m128i v25; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v26[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v27; // [rsp+50h] [rbp-80h] BYREF
  __m128i v28; // [rsp+60h] [rbp-70h] BYREF
  __m128i v29; // [rsp+70h] [rbp-60h] BYREF
  char v30; // [rsp+80h] [rbp-50h]
  void *v31; // [rsp+88h] [rbp-48h] BYREF
  __m128i *v32; // [rsp+90h] [rbp-40h]
  _QWORD v33[7]; // [rsp+98h] [rbp-38h] BYREF

  v23.m128i_i64[0] = a2;
  v23.m128i_i64[1] = a3;
  if ( a3 )
  {
    v18 = 0;
    v19 = 1;
    v20 = 0;
    while ( 1 )
    {
      v25 = 0u;
      LOBYTE(v26[0]) = 59;
      sub_232E160(&v28, &v23, v26, 1u);
      v4 = _mm_loadu_si128(&v29);
      v25 = _mm_loadu_si128(&v28);
      v23 = v4;
      if ( (unsigned __int8)sub_95CB50((const void **)&v25, "min-bits=", 9u) )
      {
        if ( sub_C93C90(v25.m128i_i64[0], v25.m128i_i64[1], 0, (unsigned __int64 *)&v28)
          || (v20 = v28.m128i_i32[0], v28.m128i_i64[0] != v28.m128i_u32[0]) )
        {
          v5 = sub_C63BB0();
          v30 = 1;
          v7 = v6;
          v32 = &v25;
          v28.m128i_i64[0] = (__int64)"invalid argument to Scalarizer pass min-bits parameter: '{0}' ";
          v29.m128i_i64[0] = (__int64)v33;
          v21 = v5;
          v28.m128i_i64[1] = 62;
          v31 = &unk_49DB108;
          v33[0] = &v31;
          v29.m128i_i64[1] = 1;
          sub_23328D0((__int64)v26, (__int64)&v28);
          sub_23058C0(&v24, (__int64)v26, v21, v7);
          v8 = v24;
          v9 = (__int64 *)v26[0];
          *(_BYTE *)(a1 + 40) |= 3u;
          *(_QWORD *)a1 = v8 & 0xFFFFFFFFFFFFFFFELL;
          if ( v9 != &v27 )
            j_j___libc_free_0((unsigned __int64)v9);
          goto LABEL_8;
        }
      }
      else
      {
        v11 = sub_95CB50((const void **)&v25, "no-", 3u) ^ 1;
        if ( v25.m128i_i64[1] == 10
          && *(_QWORD *)v25.m128i_i64[0] == 0x6F74732D64616F6CLL
          && *(_WORD *)(v25.m128i_i64[0] + 8) == 25970 )
        {
          v18 = v11;
        }
        else
        {
          v19 = v11;
          if ( !sub_9691B0((const void *)v25.m128i_i64[0], v25.m128i_u64[1], "variable-insert-extract", 23) )
          {
            v14 = sub_C63BB0();
            v30 = 1;
            v16 = v15;
            v32 = &v25;
            v28.m128i_i64[0] = (__int64)"invalid Scalarizer pass parameter '{0}' ";
            v29.m128i_i64[0] = (__int64)v33;
            v22 = v14;
            v28.m128i_i64[1] = 40;
            v31 = &unk_49DB108;
            v33[0] = &v31;
            v29.m128i_i64[1] = 1;
            sub_23328D0((__int64)v26, (__int64)&v28);
            sub_23058C0(&v24, (__int64)v26, v22, v16);
            v17 = v24;
            *(_BYTE *)(a1 + 40) |= 3u;
            *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFFELL;
            sub_2240A30(v26);
            goto LABEL_8;
          }
        }
      }
      if ( !v23.m128i_i64[1] )
      {
        v12 = v19;
        LOBYTE(a3) = v18;
        goto LABEL_14;
      }
    }
  }
  v20 = 0;
  v12 = 1;
LABEL_14:
  v13 = *(_BYTE *)(a1 + 40);
  *(_BYTE *)(a1 + 4) = v12;
  *(_BYTE *)(a1 + 5) = a3;
  *(_DWORD *)a1 = v20;
  *(_QWORD *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 40) = v13 & 0xFC | 2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
LABEL_8:
  sub_C7D6A0(0, 0, 4);
  return a1;
}
