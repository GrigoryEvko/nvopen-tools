// Function: sub_2516400
// Address: 0x2516400
//
__int64 __fastcall sub_2516400(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, char a5, int a6)
{
  __int64 *v9; // rdi
  _QWORD *v10; // r9
  __int64 *v11; // r15
  __int64 result; // rax
  unsigned int v13; // edx
  int *v14; // r13
  int *v15; // r14
  int v16; // r13d
  unsigned __int64 v17; // rdi
  __int64 *v18; // rax
  __int64 *v19; // [rsp+18h] [rbp-C8h]
  _QWORD *v20; // [rsp+20h] [rbp-C0h]
  int v22; // [rsp+2Ch] [rbp-B4h] BYREF
  char v23; // [rsp+3Eh] [rbp-A2h] BYREF
  unsigned __int8 v24; // [rsp+3Fh] [rbp-A1h] BYREF
  _QWORD v25[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v26; // [rsp+60h] [rbp-80h] BYREF
  __int64 v27; // [rsp+68h] [rbp-78h]
  _BYTE v28[112]; // [rsp+70h] [rbp-70h] BYREF

  v25[0] = &v23;
  v25[1] = &v22;
  v25[2] = &v24;
  v22 = a6;
  v23 = 0;
  v24 = 0;
  sub_250D360((__int64)&v26, a2);
  v9 = v26;
  v19 = &v26[2 * (unsigned int)v27];
  if ( v19 != v26 )
  {
    v10 = v25;
    v11 = v26;
    do
    {
      v20 = v10;
      sub_25157B0(
        a1,
        v11,
        a3,
        a4,
        (unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD *, __int64 **))sub_2506230,
        (__int64)v10);
      if ( v24 )
        break;
      v10 = v20;
      if ( a5 )
        break;
      v23 = 1;
      v11 += 2;
    }
    while ( v19 != v11 );
    v9 = v26;
  }
  if ( v9 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v9);
  result = v24;
  v13 = v24;
  if ( v24 )
  {
    v16 = v22;
    if ( !v22 )
      return result;
    goto LABEL_21;
  }
  v14 = (int *)(a3 + 4 * a4);
  v23 = 1;
  v26 = (__int64 *)v28;
  v27 = 0x600000000LL;
  if ( (int *)a3 != v14 )
  {
    v15 = (int *)a3;
    while ( !(unsigned __int8)sub_2512BA0(a1, a2, *v15, (__int64)&v26) )
    {
      if ( v14 == ++v15 )
        goto LABEL_15;
    }
    v24 = 1;
LABEL_15:
    if ( v26 != (__int64 *)v28 )
      _libc_free((unsigned __int64)v26);
    v13 = v24;
    v16 = v22;
    result = v24;
    if ( v22 )
    {
      if ( v24 )
      {
LABEL_21:
        result = v13;
        if ( v23 )
        {
          v17 = a2->m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
          if ( (a2->m128i_i64[0] & 3) == 3 )
            v17 = *(_QWORD *)(v17 + 24);
          v18 = (__int64 *)sub_BD5C60(v17);
          v26 = (__int64 *)sub_A778C0(v18, v16, 0);
          sub_2516380(a1, a2->m128i_i64, (__int64)&v26, 1, 0);
          return v24;
        }
      }
    }
  }
  return result;
}
