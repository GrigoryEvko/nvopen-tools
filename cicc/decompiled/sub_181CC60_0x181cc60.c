// Function: sub_181CC60
// Address: 0x181cc60
//
_QWORD *__fastcall sub_181CC60(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  __int64 v10; // r12
  int v11; // r15d
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  _BYTE *v15; // r12
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rsi
  _QWORD *result; // rax
  __int64 v20; // rdi
  __int64 v21; // r12
  _BYTE *v22; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int8 *v23; // [rsp+8h] [rbp-88h] BYREF
  __int64 v24[5]; // [rsp+10h] [rbp-80h] BYREF
  int v25; // [rsp+38h] [rbp-58h]
  __int64 v26; // [rsp+40h] [rbp-50h]
  __int64 v27; // [rsp+48h] [rbp-48h]

  v7 = sub_15F2050(a2);
  v8 = sub_1632FA0(v7);
  v9 = (unsigned __int64)(sub_127FA20(v8, *(_QWORD *)a2) + 7) >> 3;
  if ( v9 )
  {
    v10 = v9;
    v11 = 1;
    if ( byte_4FA9380 )
    {
      v11 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
      if ( !v11 )
        v11 = sub_15A9FE0(v8, *(_QWORD *)a2);
    }
    v12 = sub_16498A0(a2);
    v13 = *(unsigned __int8 **)(a2 + 48);
    v24[0] = 0;
    v24[3] = v12;
    v14 = *(_QWORD *)(a2 + 40);
    v24[4] = 0;
    v24[1] = v14;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v24[2] = a2 + 24;
    v23 = v13;
    if ( v13 )
    {
      sub_1623A60((__int64)&v23, (__int64)v13, 2);
      if ( v24[0] )
        sub_161E7C0((__int64)v24, v24[0]);
      v24[0] = (__int64)v23;
      if ( v23 )
        sub_1623210((__int64)&v23, v23, (__int64)v24);
    }
    v22 = (_BYTE *)sub_181B970(*a1, *(_QWORD *)(a2 - 24), v10, v11, a2, a3, a4, a5);
    v15 = v22;
    if ( byte_4FA90E0 )
    {
      v16 = sub_1819D40((_QWORD *)*a1, *(_QWORD *)(a2 - 24));
      v22 = (_BYTE *)sub_181A560((__int64 *)*a1, v22, v16, a2);
      v15 = v22;
    }
    v17 = *a1;
    if ( *(_BYTE **)(*(_QWORD *)*a1 + 200LL) != v15 )
    {
      v18 = *(_BYTE **)(v17 + 256);
      if ( v18 == *(_BYTE **)(v17 + 264) )
      {
        sub_1287830(v17 + 248, v18, &v22);
        v17 = *a1;
        v15 = v22;
      }
      else
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = v15;
          v18 = *(_BYTE **)(v17 + 256);
          v15 = v22;
        }
        *(_QWORD *)(v17 + 256) = v18 + 8;
        v17 = *a1;
      }
    }
    v23 = (unsigned __int8 *)a2;
    result = sub_176FB00(v17 + 128, (__int64 *)&v23);
    result[1] = v15;
    if ( v24[0] )
      return (_QWORD *)sub_161E7C0((__int64)v24, v24[0]);
  }
  else
  {
    v20 = *a1 + 128;
    v21 = *(_QWORD *)(*(_QWORD *)*a1 + 200LL);
    v24[0] = a2;
    result = sub_176FB00(v20, v24);
    result[1] = v21;
  }
  return result;
}
