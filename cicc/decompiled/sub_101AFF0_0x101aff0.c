// Function: sub_101AFF0
// Address: 0x101aff0
//
unsigned __int8 *__fastcall sub_101AFF0(int a1, __int64 *a2, __int64 *a3, __m128i *a4, unsigned int a5)
{
  _BYTE *v5; // rbp
  _BYTE *v6; // r12
  _BYTE *v7; // r13
  _BYTE *v8; // r14
  __int64 v9; // r14
  unsigned __int8 *result; // rax
  unsigned int v12; // r9d
  __m128i *v13; // r8
  unsigned int v14; // edi
  __int64 *v16; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v17; // [rsp-40h] [rbp-40h] BYREF
  _BYTE *v18[7]; // [rsp-38h] [rbp-38h] BYREF

  v18[6] = v5;
  v18[5] = v8;
  v9 = (__int64)a3;
  v18[4] = v7;
  v18[3] = v6;
  switch ( a1 )
  {
    case 13:
      return (unsigned __int8 *)sub_101B9B0(a2, a3, 0, 0, a4, a5);
    case 14:
      return sub_100E540(a2, a3, 0, a4, 0, 1);
    case 15:
      return (unsigned __int8 *)sub_101BE30(a2, a3, 0, 0, a4, a5);
    case 16:
      return (unsigned __int8 *)sub_10088F0(a2, a3, 0, a4, 0, 1);
    case 17:
      return (unsigned __int8 *)sub_101E3C0(a2, a3, 0, a4);
    case 18:
      v17 = a2;
      v18[0] = a3;
      result = (unsigned __int8 *)sub_FFE3E0(0x12u, (_BYTE **)&v17, v18, a4->m128i_i64);
      if ( !result )
        return sub_1009850((__int64)v17, (__int64)v18[0], 0, a4, 0, 1);
      return result;
    case 19:
      v12 = a5;
      v13 = a4;
      v14 = 19;
      return (unsigned __int8 *)sub_101A620(v14, (__int64 ***)a2, (unsigned __int8 *)a3, 0, v13, v12);
    case 20:
      if ( sub_98F660((unsigned __int8 *)a2, (unsigned __int8 *)a3, 1, 1) )
        return (unsigned __int8 *)sub_AD62B0(a2[1]);
      v12 = a5;
      v13 = a4;
      a3 = (__int64 *)v9;
      v14 = 20;
      return (unsigned __int8 *)sub_101A620(v14, (__int64 ***)a2, (unsigned __int8 *)a3, 0, v13, v12);
    case 21:
      return sub_1009F30(a2, a3, 0, a4->m128i_i64, 0, 1);
    case 22:
      return (unsigned __int8 *)sub_101AA20(0x16u, (unsigned __int8 *)a2, (unsigned __int8 *)a3, a4, a5);
    case 23:
      return (unsigned __int8 *)sub_101AF30((__int64)a2, (unsigned __int8 *)a3, a4, a5);
    case 24:
      v16 = a2;
      v17 = a3;
      result = (unsigned __int8 *)sub_FFE3E0(0x18u, (_BYTE **)&v16, (_BYTE **)&v17, a4->m128i_i64);
      if ( !result )
      {
        v18[0] = v16;
        v18[1] = v17;
        return sub_1003820((__int64 *)v18, 2, 0, (__int64)a4, 0, 1);
      }
      return result;
    case 25:
      return (unsigned __int8 *)sub_101D1E0(a2, a3, 0, 0, a4, a5);
    case 26:
      result = (unsigned __int8 *)sub_101D570(26, a2, a3, 0, a4, a5);
      if ( !result )
        return (unsigned __int8 *)sub_1006470((unsigned __int8 *)a2, v9, a4);
      return result;
    case 27:
      result = (unsigned __int8 *)sub_101D570(27, a2, a3, 0, a4, a5);
      if ( !result )
        return (unsigned __int8 *)sub_1004A20((unsigned __int8 *)a2, v9, (__int64)a4);
      return result;
    case 28:
      return (unsigned __int8 *)sub_101D750(a2, a3, a4, a5);
    case 29:
      return (unsigned __int8 *)sub_1010B00(a2, a3, a4, a5);
    case 30:
      return (unsigned __int8 *)sub_101B6D0(a2, a3, a4, a5);
    default:
      BUG();
  }
}
