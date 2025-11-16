// Function: sub_B26980
// Address: 0xb26980
//
__int64 __fastcall sub_B26980(__int64 a1, const void *a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 result; // rax
  const void *v9; // r9
  size_t v10; // r12
  __int64 v11; // r10
  unsigned __int64 *v12; // rsi
  unsigned __int64 *v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-600h]
  int v15; // [rsp+18h] [rbp-5F8h]
  unsigned __int64 *v16; // [rsp+20h] [rbp-5F0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-5E8h]
  _BYTE v18[48]; // [rsp+30h] [rbp-5E0h] BYREF
  _BYTE v19[704]; // [rsp+60h] [rbp-5B0h] BYREF
  _BYTE v20[752]; // [rsp+320h] [rbp-2F0h] BYREF

  if ( !a3 )
  {
    sub_B26290((__int64)v20, a4, a5, 0);
    sub_B24D40(a1, (__int64)v20, (__int64)v20);
    return sub_B1A8B0((__int64)v20, (__int64)v20);
  }
  v9 = a2;
  v10 = 16 * a3;
  v16 = (unsigned __int64 *)v18;
  v17 = 0x300000000LL;
  v11 = (16 * a3) >> 4;
  if ( (unsigned __int64)(16 * a3) > 0x30 )
  {
    v14 = (16 * a3) >> 4;
    sub_C8D5F0(&v16, v18, v14, 16);
    LODWORD(v11) = v14;
    v9 = a2;
    v13 = &v16[2 * (unsigned int)v17];
  }
  else
  {
    v12 = (unsigned __int64 *)v18;
    if ( !v10 )
      goto LABEL_6;
    v13 = (unsigned __int64 *)v18;
  }
  v15 = v11;
  memcpy(v13, v9, v10);
  v12 = v16;
  LODWORD(v10) = v17;
  LODWORD(v11) = v15;
LABEL_6:
  LODWORD(v17) = v11 + v10;
  sub_B18B40((__int64 *)&v16, (__m128i *)&v12[2 * (unsigned int)(v11 + v10)], (char *)a4, (char *)&a4[2 * a5]);
  sub_B26290((__int64)v19, v16, (unsigned int)v17, 1u);
  sub_B26290((__int64)v20, a4, a5, 0);
  sub_B24D40(a1, (__int64)v19, (__int64)v20);
  sub_B1A8B0((__int64)v20, (__int64)v19);
  result = sub_B1A8B0((__int64)v19, (__int64)v19);
  if ( v16 != (unsigned __int64 *)v18 )
    return _libc_free(v16, v19);
  return result;
}
