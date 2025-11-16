// Function: sub_698D30
// Address: 0x698d30
//
__int64 __fastcall sub_698D30(__int64 a1, __m128i *a2, __m128i *a3, unsigned int a4, int a5, __int64 *a6, _QWORD *a7)
{
  __int64 v8; // r12
  __int64 v10; // r13
  __int64 v11; // rdi
  unsigned __int8 *v12; // rax
  unsigned int v13; // r12d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v20; // [rsp+18h] [rbp-548h] BYREF
  __int64 v21; // [rsp+20h] [rbp-540h] BYREF
  __int64 v22; // [rsp+28h] [rbp-538h] BYREF
  __m128i v23; // [rsp+30h] [rbp-530h] BYREF
  __int64 v24; // [rsp+48h] [rbp-518h]
  _BYTE v25[160]; // [rsp+70h] [rbp-4F0h] BYREF
  __int64 v26[44]; // [rsp+110h] [rbp-450h] BYREF
  __m128i v27[4]; // [rsp+270h] [rbp-2F0h] BYREF
  _DWORD v28[71]; // [rsp+2B4h] [rbp-2ACh] BYREF
  _BYTE v29[400]; // [rsp+3D0h] [rbp-190h] BYREF

  v8 = (__int64)a2;
  sub_6E1E00(4, v25, 0, 0);
  sub_68ACF0(a1, (__int64)v26);
  v10 = v26[0];
  v11 = v26[0];
  v12 = sub_694FD0(v26[0], a2->m128i_i8, &v23);
  if ( !v12 )
  {
    a2 = a3;
    v11 = 2283;
    sub_686470(0x8EBu, a3, v8, v10);
    goto LABEL_3;
  }
  if ( (v12[84] & 2) != 0 )
  {
LABEL_3:
    v13 = 0;
    *a7 = sub_72C930(v11);
    goto LABEL_4;
  }
  a2 = 0;
  v22 = 0;
  if ( !(unsigned int)sub_84C4B0(
                        v24,
                        0,
                        0,
                        1,
                        (unsigned int)v26,
                        (unsigned int)&v22,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        4,
                        0,
                        (__int64)a3,
                        a4,
                        0,
                        0,
                        (__int64)v29,
                        (__int64)&v20)
    || (a2 = (__m128i *)v26,
        sub_7022F0(
          (unsigned int)v29,
          (unsigned int)v26,
          v20,
          1,
          0,
          0,
          0,
          0,
          (__int64)&dword_4F077C8,
          (__int64)a3,
          (__int64)&dword_4F077C8,
          (__int64)v27,
          0,
          (__int64)&v21),
        !v21) )
  {
    v11 = v22;
LABEL_7:
    sub_6E1990(v11);
    goto LABEL_3;
  }
  v15 = sub_6EB5C0(v29);
  v11 = v22;
  if ( !v15 )
    goto LABEL_7;
  sub_6E1990(v22);
  if ( a5 )
    sub_6980A0(v27, v28, a4, 0, 0, 0);
  v16 = sub_73D4C0(v27[0].m128i_i64[0], dword_4F077C4 == 2);
  v17 = sub_736020(v16, 0);
  a2 = v27;
  v13 = 1;
  *a6 = v17;
  v11 = v17;
  sub_68BC10(v17, v27);
  *a7 = v27[0].m128i_i64[0];
LABEL_4:
  sub_6E2B30(v11, a2);
  return v13;
}
