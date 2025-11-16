// Function: sub_118E070
// Address: 0x118e070
//
__int64 __fastcall sub_118E070(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rsi
  unsigned __int8 *v14; // rbx
  __int64 v15; // r15
  __int64 *v17; // rsi
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 **v20; // rbx
  unsigned __int8 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int8 **v27; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v28; // [rsp+18h] [rbp-C8h]
  char v29; // [rsp+27h] [rbp-B9h] BYREF
  unsigned __int8 *v30; // [rsp+28h] [rbp-B8h] BYREF
  unsigned __int8 *v31; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 *v32; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v33[6]; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int8 **v34; // [rsp+70h] [rbp-70h] BYREF
  __int64 v35; // [rsp+78h] [rbp-68h]
  _BYTE v36[96]; // [rsp+80h] [rbp-60h] BYREF

  v7 = *(unsigned __int8 **)(a2 - 64);
  v8 = *(unsigned __int8 **)(a2 - 32);
  v29 = 0;
  v30 = v7;
  if ( (unsigned __int8)sub_B52A20(a3, 1, a3, a4, a5) )
  {
    v12 = v30;
    v29 = 1;
    v30 = v8;
    v28 = v12;
  }
  else
  {
    v28 = v8;
    if ( !(unsigned __int8)sub_B52A20(a3, 0, v9, v10, v11) )
      return 0;
  }
  v13 = *(unsigned __int8 **)(a3 - 64);
  v14 = *(unsigned __int8 **)(a3 - 32);
  v33[0] = &v30;
  v33[3] = &v29;
  v31 = v13;
  v33[1] = a1;
  v33[2] = a2;
  v33[4] = &v31;
  v15 = sub_118DDB0((__int64)v33, v13, v14);
  if ( v15 )
    return v15;
  v15 = sub_118DDB0((__int64)v33, v14, v31);
  if ( v15 )
    return v15;
  if ( *v28 <= 0x1Cu )
    return 0;
  v17 = (__int64 *)v31;
  v34 = (unsigned __int8 **)v36;
  v35 = 0x600000000LL;
  v18 = sub_101E970(v28, (__int64)v31, (__int64)v14, a1 + 6, 0, (__int64)&v34);
  if ( v30 == v18
    || (v17 = (__int64 *)v14, v19 = sub_101E970(v28, (__int64)v14, (__int64)v31, a1 + 6, 0, (__int64)&v34), v30 == v19) )
  {
    v20 = v34;
    v27 = &v34[(unsigned int)v35];
    if ( v27 != v34 )
    {
      do
      {
        v21 = *v20++;
        sub_B44F30(v21);
        sub_B44B50((__int64 *)v21, (__int64)v17);
        sub_B44A60((__int64)v21);
        v22 = a1[2].m128i_i64[1];
        v17 = (__int64 *)&v32;
        v32 = v21;
        sub_1187E30(v22 + 2096, (__int64 *)&v32, v23, v24, v25, v26);
      }
      while ( v27 != v20 );
    }
    v17 = (__int64 *)a2;
    v15 = (__int64)sub_F162A0((__int64)a1, a2, (__int64)v28);
  }
  if ( v34 != (unsigned __int8 **)v36 )
    _libc_free(v34, v17);
  return v15;
}
