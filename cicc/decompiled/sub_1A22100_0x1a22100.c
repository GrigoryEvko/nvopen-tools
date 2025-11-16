// Function: sub_1A22100
// Address: 0x1a22100
//
__int64 ***__fastcall sub_1A22100(__int64 a1, __int64 ***a2, int a3, double a4, double a5, double a6)
{
  __int64 ***result; // rax
  __int64 *v7; // r14
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 **v10; // rdi
  __int64 **v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  __m128i v18; // [rsp-88h] [rbp-88h] BYREF
  char v19; // [rsp-78h] [rbp-78h]
  char v20; // [rsp-77h] [rbp-77h]
  __m128i v21; // [rsp-68h] [rbp-68h] BYREF
  char v22; // [rsp-58h] [rbp-58h]
  char v23; // [rsp-57h] [rbp-57h]
  _BYTE v24[16]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v25; // [rsp-38h] [rbp-38h]

  result = a2;
  if ( a3 != 1 )
  {
    v7 = (__int64 *)(a1 + 192);
    v8 = (__int64)a2;
    v9 = sub_1644C60(**a2, 8 * a3);
    v10 = *a2;
    v23 = 1;
    v11 = (__int64 **)v9;
    v22 = 3;
    v21.m128i_i64[0] = (__int64)"isplat";
    v12 = sub_15A04A0(v10);
    v13 = sub_15A3CB0(v12, v11, 0);
    v14 = (__int64 *)sub_15A04A0(v11);
    v20 = 1;
    v15 = sub_15A2C70(v14, v13, 0, a4, a5, a6);
    v19 = 3;
    v18.m128i_i64[0] = (__int64)"zext";
    if ( v11 != *a2 )
    {
      if ( *((_BYTE *)a2 + 16) > 0x10u )
      {
        v25 = 257;
        v17 = (_QWORD *)sub_15FDBD0(37, (__int64)a2, (__int64)v11, (__int64)v24, 0);
        v8 = (__int64)sub_1A1C7B0(v7, v17, &v18);
        if ( *(_BYTE *)(v8 + 16) <= 0x10u )
        {
LABEL_6:
          if ( *(_BYTE *)(v15 + 16) <= 0x10u )
            return (__int64 ***)sub_15A2C20((__int64 *)v8, v15, 0, 0, a4, a5, a6);
        }
LABEL_8:
        v25 = 257;
        v16 = (_QWORD *)sub_15FB440(15, (__int64 *)v8, v15, (__int64)v24, 0);
        return (__int64 ***)sub_1A1C7B0(v7, v16, &v21);
      }
      v8 = sub_15A46C0(37, a2, v11, 0);
    }
    if ( *(_BYTE *)(v8 + 16) <= 0x10u )
      goto LABEL_6;
    goto LABEL_8;
  }
  return result;
}
