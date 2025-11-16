// Function: sub_38EDF20
// Address: 0x38edf20
//
__int64 __fastcall sub_38EDF20(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned int v4; // r13d
  _DWORD *v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rsi
  void *v9; // rax
  __int64 v10; // r15
  void *v11; // rax
  _DWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // [rsp+18h] [rbp-E8h]
  __int64 v15; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-D0h] BYREF
  char v18; // [rsp+40h] [rbp-C0h]
  char v19; // [rsp+41h] [rbp-BFh]
  _QWORD v20[2]; // [rsp+50h] [rbp-B0h] BYREF
  char v21; // [rsp+60h] [rbp-A0h]
  char v22; // [rsp+61h] [rbp-9Fh]
  _QWORD *v23; // [rsp+70h] [rbp-90h] BYREF
  __int64 v24; // [rsp+78h] [rbp-88h]
  _BYTE v25[16]; // [rsp+80h] [rbp-80h] BYREF
  __m128i src; // [rsp+90h] [rbp-70h] BYREF
  _QWORD v27[2]; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v28; // [rsp+B0h] [rbp-50h] BYREF
  _QWORD v29[8]; // [rsp+C0h] [rbp-40h] BYREF

  v2 = sub_3909460(a1);
  v3 = sub_39092A0(v2);
  v25[0] = 0;
  v14 = v3;
  v23 = v25;
  v24 = 0;
  src.m128i_i64[0] = (__int64)v27;
  src.m128i_i64[1] = 0;
  LOBYTE(v27[0]) = 0;
  v16 = 0;
  v19 = 1;
  v17[0] = "expected file number in '.cv_file' directive";
  v18 = 3;
  if ( (unsigned __int8)sub_3909D40(a1, &v15, v17) )
    goto LABEL_2;
  v20[0] = "file number less than one";
  v22 = 1;
  v21 = 3;
  if ( (unsigned __int8)sub_3909C80(a1, v15 <= 0, v14, v20) )
    goto LABEL_2;
  v28.m128i_i64[0] = (__int64)"unexpected token in '.cv_file' directive";
  LOWORD(v29[0]) = 259;
  v6 = (_DWORD *)sub_3909460(a1);
  if ( (unsigned __int8)sub_3909CB0(a1, *v6 != 3, &v28) )
    goto LABEL_2;
  v4 = sub_38ECF20(a1, (unsigned __int64 *)&v23);
  if ( (_BYTE)v4 )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_3909EB0(a1, 9) )
  {
    v19 = 1;
    v17[0] = "unexpected token in '.cv_file' directive";
    v18 = 3;
    v12 = (_DWORD *)sub_3909460(a1);
    if ( (unsigned __int8)sub_3909CB0(a1, *v12 != 3, v17)
      || (unsigned __int8)sub_38ECF20(a1, (unsigned __int64 *)&src)
      || (v22 = 1,
          v20[0] = "expected checksum kind in '.cv_file' directive",
          v21 = 3,
          (unsigned __int8)sub_3909D40(a1, &v16, v20))
      || (v28.m128i_i64[0] = (__int64)"unexpected token in '.cv_file' directive",
          LOWORD(v29[0]) = 259,
          (unsigned __int8)sub_3909E20(a1, 9, &v28)) )
    {
LABEL_2:
      v4 = 1;
      goto LABEL_3;
    }
  }
  sub_38E7EA0(&v28, (char *)src.m128i_i64[0], src.m128i_i64[1]);
  v7 = (_BYTE *)src.m128i_i64[0];
  if ( (_QWORD *)v28.m128i_i64[0] == v29 )
  {
    v13 = v28.m128i_i64[1];
    if ( v28.m128i_i64[1] )
    {
      if ( v28.m128i_i64[1] == 1 )
        *(_BYTE *)src.m128i_i64[0] = v29[0];
      else
        memcpy((void *)src.m128i_i64[0], v29, v28.m128i_u64[1]);
      v13 = v28.m128i_i64[1];
      v7 = (_BYTE *)src.m128i_i64[0];
    }
    src.m128i_i64[1] = v13;
    v7[v13] = 0;
    v7 = (_BYTE *)v28.m128i_i64[0];
    goto LABEL_16;
  }
  if ( (_QWORD *)src.m128i_i64[0] == v27 )
  {
    src = v28;
    v27[0] = v29[0];
    goto LABEL_31;
  }
  v8 = v27[0];
  src = v28;
  v27[0] = v29[0];
  if ( !v7 )
  {
LABEL_31:
    v28.m128i_i64[0] = (__int64)v29;
    v7 = v29;
    goto LABEL_16;
  }
  v28.m128i_i64[0] = (__int64)v7;
  v29[0] = v8;
LABEL_16:
  v28.m128i_i64[1] = 0;
  *v7 = 0;
  if ( (_QWORD *)v28.m128i_i64[0] != v29 )
    j_j___libc_free_0(v28.m128i_u64[0]);
  v9 = (void *)sub_145CBF0((__int64 *)(*(_QWORD *)(a1 + 320) + 48LL), src.m128i_u32[2], 1);
  v10 = src.m128i_i64[1];
  v11 = memcpy(v9, (const void *)src.m128i_i64[0], src.m128i_u64[1]);
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD *, __int64, void *, __int64, _QWORD))(**(_QWORD **)(a1 + 328) + 600LL))(
          *(_QWORD *)(a1 + 328),
          (unsigned int)v15,
          v23,
          v24,
          v11,
          v10,
          (unsigned __int8)v16) )
  {
    v28.m128i_i64[0] = (__int64)"file number already allocated";
    LOWORD(v29[0]) = 259;
    v4 = sub_3909790(a1, v14, &v28, 0, 0);
  }
LABEL_3:
  if ( (_QWORD *)src.m128i_i64[0] != v27 )
    j_j___libc_free_0(src.m128i_u64[0]);
  if ( v23 != (_QWORD *)v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  return v4;
}
