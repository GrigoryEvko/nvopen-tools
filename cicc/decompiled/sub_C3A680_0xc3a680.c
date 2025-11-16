// Function: sub_C3A680
// Address: 0xc3a680
//
__int64 __fastcall sub_C3A680(__int64 a1, __int64 *a2)
{
  const __m128i *v2; // roff
  __int64 v3; // rdx
  __int64 v5; // rcx
  bool v6; // [rsp+1Fh] [rbp-D1h] BYREF
  _QWORD *v7; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD *v8; // [rsp+28h] [rbp-C8h]
  _QWORD *v9; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v10; // [rsp+38h] [rbp-B8h]
  __int64 v11[4]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v12[2]; // [rsp+60h] [rbp-90h] BYREF
  char v13; // [rsp+74h] [rbp-7Ch]
  _QWORD *v14; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v15; // [rsp+88h] [rbp-68h]
  __m128i v16; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+B0h] [rbp-40h]
  __int32 v18; // [rsp+B8h] [rbp-38h]

  v2 = (const __m128i *)*a2;
  v16 = _mm_loadu_si128((const __m128i *)*a2);
  v3 = v2[1].m128i_i64[0];
  v16.m128i_i32[1] = -1022;
  v17 = v3;
  v18 = v2[1].m128i_i32[2];
  sub_C33EB0(v11, a2);
  sub_C396A0((__int64)v11, &v16, 1, &v6);
  sub_C33EB0(v12, v11);
  sub_C396A0((__int64)v12, dword_3F657A0, 1, &v6);
  sub_C349F0((__int64)&v14, (__int64)v12);
  if ( v15 <= 0x40 )
  {
    v7 = v14;
  }
  else
  {
    v7 = (_QWORD *)*v14;
    j_j___libc_free_0_0(v14);
  }
  if ( (v13 & 6) != 0 && (v13 & 7) != 3 && v6 )
  {
    sub_C396A0((__int64)v12, &v16, 1, &v6);
    sub_C33EB0(&v14, v11);
    sub_C3B1F0(&v14, v12, 1, v5);
    sub_C396A0((__int64)&v14, dword_3F657A0, 1, &v6);
    sub_C349F0((__int64)&v9, (__int64)&v14);
    if ( v10 <= 0x40 )
    {
      v8 = v9;
    }
    else
    {
      v8 = (_QWORD *)*v9;
      j_j___libc_free_0_0(v9);
    }
    sub_C338F0((__int64)&v14);
  }
  else
  {
    v8 = 0;
  }
  sub_C438C0(a1, 128, &v7, 2);
  sub_C338F0((__int64)v12);
  sub_C338F0((__int64)v11);
  return a1;
}
