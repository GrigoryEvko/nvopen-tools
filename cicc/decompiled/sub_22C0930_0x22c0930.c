// Function: sub_22C0930
// Address: 0x22c0930
//
_WORD *__fastcall sub_22C0930(_WORD *a1, char *a2, __int64 *a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int8 *v9; // rsi
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v14; // r8
  __int64 *v15; // rdx
  const void *v16; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-88h]
  __m128i v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+20h] [rbp-70h] BYREF
  __int64 v20; // [rsp+28h] [rbp-68h]
  __int64 v21; // [rsp+30h] [rbp-60h]
  __int64 v22; // [rsp+38h] [rbp-58h]
  __int64 v23; // [rsp+40h] [rbp-50h]
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int16 v25; // [rsp+50h] [rbp-40h]

  v9 = (unsigned __int8 *)sub_AD6220(a3[1], a4);
  v10 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    goto LABEL_10;
  if ( (unsigned int)v10 - 67 <= 0xC )
  {
    v11 = *((_QWORD *)a2 + 1);
    v18 = (__m128i)a5;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 257;
    v12 = sub_1002A60((unsigned int)v10 - 29, v9, v11, v18.m128i_i64);
    if ( v12 )
      goto LABEL_4;
LABEL_10:
    *a1 = 6;
    return a1;
  }
  if ( (unsigned int)v10 - 42 > 0x11 )
  {
    if ( v10 == 96 )
    {
      v17 = *(_DWORD *)(a4 + 8);
      if ( v17 > 0x40 )
        sub_C43780((__int64)&v16, (const void **)a4);
      else
        v16 = *(const void **)a4;
      goto LABEL_7;
    }
    goto LABEL_10;
  }
  v14 = (__int64 *)*((_QWORD *)a2 - 8);
  v15 = (__int64 *)*((_QWORD *)a2 - 4);
  v18 = (__m128i)a5;
  v25 = 257;
  if ( a3 == v14 )
    v14 = (__int64 *)v9;
  v19 = 0;
  if ( a3 == v15 )
    v15 = (__int64 *)v9;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v12 = (__int64)sub_101E7C0((unsigned int)v10 - 29, v14, v15, &v18);
  if ( !v12 )
    goto LABEL_10;
LABEL_4:
  if ( *(_BYTE *)v12 != 17 )
    goto LABEL_10;
  v17 = *(_DWORD *)(v12 + 32);
  if ( v17 > 0x40 )
    sub_C43780((__int64)&v16, (const void **)(v12 + 24));
  else
    v16 = *(const void **)(v12 + 24);
LABEL_7:
  sub_AADBC0((__int64)&v18, (__int64 *)&v16);
  sub_22C06B0((__int64)a1, (__int64)&v18, 0);
  sub_969240(&v19);
  sub_969240(v18.m128i_i64);
  sub_969240((__int64 *)&v16);
  return a1;
}
