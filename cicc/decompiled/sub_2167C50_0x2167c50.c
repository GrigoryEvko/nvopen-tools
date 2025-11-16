// Function: sub_2167C50
// Address: 0x2167c50
//
__int64 __fastcall sub_2167C50(
        __int64 a1,
        int a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        int a6,
        __int128 a7,
        int a8,
        int *a9,
        int a10,
        char a11)
{
  char v12; // r15
  char **v13; // rax
  char *v14; // rsi
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  __int64 v17; // r9
  bool v18; // zf
  int v22; // [rsp+20h] [rbp-180h]
  int v23; // [rsp+24h] [rbp-17Ch]
  __m128i v25; // [rsp+30h] [rbp-170h]
  __int64 v28; // [rsp+60h] [rbp-140h]
  _QWORD *v30; // [rsp+70h] [rbp-130h] BYREF
  __int64 v31; // [rsp+78h] [rbp-128h]
  _QWORD v32[2]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v33[2]; // [rsp+90h] [rbp-110h] BYREF
  _QWORD v34[22]; // [rsp+A0h] [rbp-100h] BYREF
  char v35; // [rsp+156h] [rbp-4Ah]

  v25 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = a11;
  v22 = a10;
  v23 = 0;
  if ( *((_BYTE *)a9 + 4) )
    v23 = *a9;
  sub_2164E40(v33, a3, (int)a4, a5, v25.m128i_i32[0], v25.m128i_i32[2]);
  if ( !v12 )
  {
    v13 = off_4CD4938;
LABEL_5:
    v14 = *v13;
    goto LABEL_6;
  }
  if ( (v35 & 0x20) != 0 )
  {
    v13 = off_4CD4940;
    goto LABEL_5;
  }
  v14 = off_4CD4948[0];
LABEL_6:
  v15 = -1;
  v30 = v32;
  if ( v14 )
    v15 = (__int64)&v14[strlen(v14)];
  sub_2165CE0((__int64 *)&v30, v14, v15);
  v33[0] = (__int64)&unk_4A027E0;
  sub_39BA210(v33);
  sub_39B3060(a1, a2, (_DWORD)v30, v31, a3, a6, (__int64)a4, a5, v25.m128i_i64[0], v25.m128i_i64[1], 1, v23, v22);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  *(_BYTE *)(a1 + 936) = v12;
  *(_QWORD *)a1 = &unk_4A02A78;
  *(_BYTE *)(a1 + 937) = byte_4FD2780;
  v16 = (_QWORD *)sub_22077B0(800);
  if ( v16 )
  {
    memset(v16, 0, 0x320u);
    *((_BYTE *)v16 + 777) = 1;
    v16[88] = v16 + 90;
    *v16 = &unk_4A03C48;
  }
  *(_QWORD *)(a1 + 944) = v16;
  v17 = a1 + 960;
  if ( v25.m128i_i64[0] )
  {
    v33[0] = (__int64)v34;
    sub_2165CE0(v33, v25.m128i_i64[0], v25.m128i_i64[0] + v25.m128i_i64[1]);
    v17 = a1 + 960;
  }
  else
  {
    LOBYTE(v34[0]) = 0;
    v33[0] = (__int64)v34;
    v33[1] = 0;
  }
  if ( a4 )
  {
    v28 = v17;
    v30 = v32;
    sub_2165CE0((__int64 *)&v30, a4, (__int64)&a4[a5]);
    v17 = v28;
  }
  else
  {
    LOBYTE(v32[0]) = 0;
    v31 = 0;
    v30 = v32;
  }
  sub_21651F0(v17, a3, (__int64)&v30, v33, a1);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  if ( (_QWORD *)v33[0] != v34 )
    j_j___libc_free_0(v33[0], v34[0] + 1LL);
  *(_QWORD *)(a1 + 83288) = a1 + 83304;
  *(_QWORD *)(a1 + 83296) = 0x800000000LL;
  v18 = byte_4FD2860 == 0;
  *(_DWORD *)(a1 + 952) = *(_DWORD *)(a3 + 44) != 23;
  if ( v18 )
    *(_BYTE *)(a1 + 640) |= 1u;
  return sub_39B2D50(a1);
}
