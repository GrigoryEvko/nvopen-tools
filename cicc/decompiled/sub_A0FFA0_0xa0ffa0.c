// Function: sub_A0FFA0
// Address: 0xa0ffa0
//
__int64 __fastcall sub_A0FFA0(__m128i *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  const char *v14; // rax
  char v15; // dl
  unsigned __int64 v16; // rax
  __int64 v17; // [rsp-10h] [rbp-350h]
  unsigned __int32 v19; // [rsp+18h] [rbp-328h]
  unsigned int v20; // [rsp+1Ch] [rbp-324h] BYREF
  __m128i v21; // [rsp+20h] [rbp-320h] BYREF
  __int64 v22; // [rsp+30h] [rbp-310h] BYREF
  unsigned __int64 v23; // [rsp+38h] [rbp-308h]
  unsigned __int64 v24; // [rsp+40h] [rbp-300h] BYREF
  unsigned __int8 v25; // [rsp+48h] [rbp-2F8h]
  _BYTE v26[32]; // [rsp+50h] [rbp-2F0h] BYREF
  __m128i v27[2]; // [rsp+70h] [rbp-2D0h] BYREF
  __int16 v28; // [rsp+90h] [rbp-2B0h]
  __m128i v29[2]; // [rsp+A0h] [rbp-2A0h] BYREF
  char v30; // [rsp+C0h] [rbp-280h]
  char v31; // [rsp+C1h] [rbp-27Fh]
  __m128i v32[3]; // [rsp+D0h] [rbp-270h] BYREF
  __int64 *v33[2]; // [rsp+100h] [rbp-240h] BYREF
  _BYTE v34[560]; // [rsp+110h] [rbp-230h] BYREF

  v20 = a2;
  if ( a2 < a1->m128i_i32[2] )
  {
    v4 = *(_QWORD *)(a1->m128i_i64[0] + 8LL * a2);
    if ( v4 )
    {
      result = *(_BYTE *)(v4 + 1) & 0x7F;
      if ( (_BYTE)result != 2 )
        return result;
    }
  }
  v22 = 0;
  v33[1] = (__int64 *)0x4000000000LL;
  v6 = a1[45].m128i_i64[0] - a1[44].m128i_i64[1];
  v33[0] = (__int64 *)v34;
  v23 = 0;
  sub_9CDFE0(&v21.m128i_i64[1], (__int64)a1[23].m128i_i64, *(_QWORD *)(a1[46].m128i_i64[0] + 8 * (a2 - (v6 >> 4))), a4);
  v8 = v21.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v21.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v21.m128i_i64[1] = 0;
    v24 = v8 | 1;
    sub_C64870(v26, &v24);
    v27[0].m128i_i64[0] = (__int64)v26;
    v28 = 260;
    v31 = 1;
    v29[0].m128i_i64[0] = (__int64)"lazyLoadOneMetadata failed jumping: ";
    v30 = 3;
    goto LABEL_7;
  }
  sub_9CEFB0((__int64)v32, (__int64)a1[23].m128i_i64, 0, v7);
  if ( (v32[0].m128i_i8[8] & 1) != 0 )
  {
    v32[0].m128i_i8[8] &= ~2u;
    v12 = v32[0].m128i_i64[0];
    v32[0].m128i_i64[0] = 0;
    v21.m128i_i64[1] = v12 | 1;
    v13 = v12 & 0xFFFFFFFFFFFFFFFELL;
    if ( v13 )
    {
      v24 = v13 | 1;
      sub_C64870(v26, &v24);
      v27[0].m128i_i64[0] = (__int64)v26;
      v14 = "lazyLoadOneMetadata failed advanceSkippingSubblocks: ";
      v28 = 260;
      v31 = 1;
LABEL_11:
      v29[0].m128i_i64[0] = (__int64)v14;
      v30 = 3;
LABEL_7:
      sub_9C6370(v32, v29, v27, v9, v10, v11);
      sub_C64D30(v32, 1);
    }
  }
  else
  {
    v21.m128i_i64[1] = 1;
    v19 = v32[0].m128i_u32[1];
  }
  sub_A4B600(&v24, &a1[23], v19, v33, &v22);
  v15 = v25 & 1;
  v25 = (2 * (v25 & 1)) | v25 & 0xFD;
  if ( v15 )
  {
    sub_9C8CD0(&v21.m128i_i64[1], (__int64 *)&v24);
    sub_C64870(v26, &v21.m128i_u64[1]);
    v27[0].m128i_i64[0] = (__int64)v26;
    v14 = "Can't lazyload MD: ";
    v28 = 260;
    v31 = 1;
    goto LABEL_11;
  }
  sub_A09F80(&v21, a1, v33, (_BYTE *)(unsigned int)v24, a3, &v20, v22, v23);
  if ( (v21.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v16 = v21.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    v21.m128i_i64[0] = 0;
    v21.m128i_i64[1] = v16;
    sub_C64870(v26, &v21.m128i_u64[1]);
    v27[0].m128i_i64[0] = (__int64)v26;
    v28 = 260;
    v14 = "Can't lazyload MD, parseOneMetadata: ";
    v31 = 1;
    goto LABEL_11;
  }
  result = v25;
  if ( (v25 & 2) != 0 )
    sub_9CE230(&v24);
  if ( (v25 & 1) != 0 && v24 )
    result = (*(__int64 (__fastcall **)(unsigned __int64, __m128i *, __int64))(*(_QWORD *)v24 + 8LL))(v24, a1, v17);
  if ( (_BYTE *)v33[0] != v34 )
    return _libc_free(v33[0], a1);
  return result;
}
