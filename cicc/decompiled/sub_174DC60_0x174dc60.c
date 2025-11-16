// Function: sub_174DC60
// Address: 0x174dc60
//
_QWORD *__fastcall sub_174DC60(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 v12; // rcx
  char v13; // al
  __int64 **v14; // rsi
  __int64 *v15; // r14
  __int64 **v16; // r12
  __int64 v17; // r13
  int v18; // esi
  int v19; // eax
  unsigned int v20; // eax
  _QWORD *result; // rax
  __int64 v22; // r12
  __int64 v23; // rbx
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 *v27; // [rsp+8h] [rbp-68h]
  unsigned int v28; // [rsp+10h] [rbp-60h]
  bool v29; // [rsp+17h] [rbp-59h]
  bool v30; // [rsp+18h] [rbp-58h]
  _QWORD *v31; // [rsp+18h] [rbp-58h]
  _QWORD *v32; // [rsp+18h] [rbp-58h]
  _QWORD *v33; // [rsp+18h] [rbp-58h]
  _QWORD *v34; // [rsp+18h] [rbp-58h]
  _BYTE v35[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v36; // [rsp+30h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v11 = *(_QWORD *)(a2 - 8);
  else
    v11 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v12 = *(_QWORD *)v11;
  v13 = *(_BYTE *)(*(_QWORD *)v11 + 16LL);
  if ( (unsigned __int8)(v13 - 65) > 1u )
    return 0;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
    v14 = *(__int64 ***)(v12 - 8);
  else
    v14 = (__int64 **)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
  v15 = *v14;
  v16 = *(__int64 ***)a2;
  v27 = (__int64 *)v12;
  v29 = v13 == 66;
  v17 = **v14;
  v30 = *(_BYTE *)(a2 + 16) == 64;
  v18 = sub_16431D0(v17) - (v13 == 66);
  v19 = sub_16431D0((__int64)v16) - v30;
  if ( v19 <= v18 )
    v18 = v19;
  if ( (int)sub_16431F0(*v27) < v18 )
    return 0;
  v28 = sub_16431D0((__int64)v16);
  v20 = sub_16431D0(v17);
  if ( v28 > v20 )
  {
    if ( v30 && v29 )
    {
      v36 = 257;
      result = sub_1648A60(56, 1u);
      if ( result )
      {
        v31 = result;
        sub_15FC810((__int64)result, (__int64)v15, (__int64)v16, (__int64)v35, 0);
        return v31;
      }
    }
    else
    {
      v36 = 257;
      result = sub_1648A60(56, 1u);
      if ( result )
      {
        v33 = result;
        sub_15FC690((__int64)result, (__int64)v15, (__int64)v16, (__int64)v35, 0);
        return v33;
      }
    }
    return result;
  }
  if ( v28 >= v20 )
  {
    if ( (__int64 **)v17 != v16 )
    {
      v36 = 257;
      result = sub_1648A60(56, 1u);
      if ( result )
      {
        v32 = result;
        sub_15FD590((__int64)result, (__int64)v15, (__int64)v16, (__int64)v35, 0);
        return v32;
      }
      return result;
    }
    v22 = *(_QWORD *)(a2 + 8);
    if ( v22 )
    {
      v23 = *a1;
      do
      {
        v24 = sub_1648700(v22);
        sub_170B990(v23, (__int64)v24);
        v22 = *(_QWORD *)(v22 + 8);
      }
      while ( v22 );
      if ( (__int64 *)a2 == v15 )
        v15 = (__int64 *)sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, (__int64)v15, a3, a4, a5, a6, v25, v26, a9, a10);
      return (_QWORD *)a2;
    }
    return 0;
  }
  v36 = 257;
  result = sub_1648A60(56, 1u);
  if ( result )
  {
    v34 = result;
    sub_15FC510((__int64)result, (__int64)v15, (__int64)v16, (__int64)v35, 0);
    return v34;
  }
  return result;
}
