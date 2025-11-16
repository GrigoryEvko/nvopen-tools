// Function: sub_1F6DC60
// Address: 0x1f6dc60
//
__int64 *__fastcall sub_1F6DC60(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 *a6,
        __m128i a7,
        double a8,
        __m128i a9,
        _DWORD *a10)
{
  __int64 v10; // r11
  __int64 *v11; // r10
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned int v14; // ebx
  bool v15; // dl
  unsigned int v16; // eax
  __int64 v17; // rax
  unsigned int v18; // edx
  bool v20; // al
  __int64 v21; // rax
  unsigned int v22; // edx
  __int128 v23; // [rsp-10h] [rbp-A0h]
  const void **v25; // [rsp+10h] [rbp-80h]
  const void **v26; // [rsp+18h] [rbp-78h]
  const void **v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  __int64 *v31; // [rsp+28h] [rbp-68h]
  bool v32; // [rsp+28h] [rbp-68h]
  __int64 *v33; // [rsp+28h] [rbp-68h]
  _QWORD v34[8]; // [rsp+50h] [rbp-40h] BYREF

  v10 = a3;
  v11 = a6;
  v12 = 0;
  v13 = 0;
  v14 = a4;
  v34[0] = a4;
  v34[1] = a5;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) > 0x5Fu )
    {
      v15 = (unsigned __int8)(a4 - 86) <= 0x17u || (unsigned __int8)(a4 - 8) <= 5u;
      goto LABEL_4;
    }
  }
  else
  {
    v25 = a5;
    v32 = sub_1F58CD0((__int64)v34);
    v20 = sub_1F58D20((__int64)v34);
    v15 = v32;
    v10 = a3;
    a5 = v25;
    v11 = a6;
    if ( !v20 )
    {
LABEL_4:
      if ( v15 )
        v16 = a10[16];
      else
        v16 = a10[15];
      if ( v16 <= 1 )
        goto LABEL_7;
      goto LABEL_11;
    }
  }
  v16 = a10[17];
  if ( v16 <= 1 )
  {
LABEL_7:
    v26 = a5;
    v29 = v10;
    v31 = v11;
    v17 = sub_1D38BB0((__int64)v11, 1, v10, v14, a5, 0, a7, a8, a9, 0);
    v11 = v31;
    v10 = v29;
    v12 = v17;
    a5 = v26;
    v13 = v18;
    goto LABEL_8;
  }
LABEL_11:
  if ( v16 == 2 )
  {
    v28 = a5;
    v30 = v10;
    v33 = v11;
    v21 = sub_1D38BB0((__int64)v11, -1, v10, v14, a5, 0, a7, a8, a9, 0);
    a5 = v28;
    v10 = v30;
    v12 = v21;
    v11 = v33;
    v13 = v22;
  }
LABEL_8:
  *((_QWORD *)&v23 + 1) = v13;
  *(_QWORD *)&v23 = v12;
  return sub_1D332F0(v11, 120, v10, v14, a5, 0, *(double *)a7.m128i_i64, a8, a9, a1, a2, v23);
}
