// Function: sub_2588B60
// Address: 0x2588b60
//
__int64 __fastcall sub_2588B60(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rax
  _BYTE *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int *v12; // r13
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 v14; // rdi
  bool (__fastcall *v15)(__int64); // rax
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rax
  _QWORD *(__fastcall *v18)(__m128i *, __int64); // rax
  _QWORD *v19; // rdx
  _QWORD *v20; // rax
  unsigned __int64 v21; // r12
  __int64 v22; // r12
  bool v23; // zf
  __m128i v24; // xmm1
  __int64 v25; // rax
  __int64 v26; // [rsp-8h] [rbp-A8h]
  bool v27; // [rsp+Fh] [rbp-91h] BYREF
  _BYTE v28[48]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v29; // [rsp+40h] [rbp-60h]
  __m128i v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+60h] [rbp-40h]

  v4 = (_BYTE *)sub_2509740((_QWORD *)(a1 + 72));
  if ( *v4 != 60 )
    goto LABEL_2;
  v6 = v4;
  if ( !(unsigned __int8)sub_2588040(a2, a1, (__int64 *)(a1 + 72), 1, &v27, 0, 0) )
    goto LABEL_2;
  v7 = *(_QWORD *)(a1 + 72);
  v8 = sub_252A820(a2, v7, *(_QWORD *)(a1 + 80), a1, 0, 0, 1);
  v11 = v26;
  v12 = (unsigned int *)v8;
  if ( !v8 )
    goto LABEL_2;
  v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 48LL);
  v14 = v13 == sub_2534F10
      ? (__int64)(v12 + 22)
      : ((__int64 (__fastcall *)(unsigned int *, __int64, __int64, __int64))v13)(v12, v7, v9, v10);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v14 + 16LL))(
          v14,
          v7,
          v9,
          v10,
          v11) )
    goto LABEL_2;
  v15 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v12 + 136LL);
  if ( v15 == sub_2534CF0 )
  {
    if ( *((_QWORD *)v12 + 47) || v12[74] )
      goto LABEL_2;
  }
  else if ( v15((__int64)v12) )
  {
    goto LABEL_2;
  }
  if ( *v6 != 60 )
    goto LABEL_2;
  sub_B4CED0((__int64)v28, (__int64)v6, *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL));
  if ( !v28[16] || !sub_CA1930(v28) )
    goto LABEL_2;
  v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 128LL);
  v17 = v16 == sub_2534CE0 ? v12[60] : v16((__int64)v12);
  if ( v17 > 1 )
    goto LABEL_2;
  if ( v17 )
  {
    v18 = *(_QWORD *(__fastcall **)(__m128i *, __int64))(*(_QWORD *)v12 + 112LL);
    if ( (char *)v18 != (char *)sub_253BEA0 )
    {
      v18(&v30, (__int64)v12);
      v20 = (_QWORD *)v31;
      goto LABEL_20;
    }
    v19 = (_QWORD *)*((_QWORD *)v12 + 29);
    v20 = &v19[12 * v12[62]];
    if ( v12[60] && v20 != v19 )
    {
      do
      {
        if ( *v19 == 0x7FFFFFFFFFFFFFFFLL )
        {
          if ( v19[1] != 0x7FFFFFFFFFFFFFFFLL )
            goto LABEL_33;
        }
        else if ( *v19 != 0x7FFFFFFFFFFFFFFELL || v19[1] != 0x7FFFFFFFFFFFFFFELL )
        {
LABEL_33:
          v20 = v19;
          break;
        }
        v19 += 12;
      }
      while ( v20 != v19 );
    }
LABEL_20:
    if ( !*v20 )
    {
      v21 = v20[1];
      if ( sub_CA1930(v28) > v21 )
      {
        v22 = 8 * v21;
        if ( !*(_BYTE *)(a1 + 120) || (v25 = *(_QWORD *)(a1 + 104), v25 == -1) || v22 != v25 || *(_BYTE *)(a1 + 112) )
        {
          *(_QWORD *)(a1 + 104) = v22;
          *(_BYTE *)(a1 + 112) = 0;
          *(_BYTE *)(a1 + 120) = 1;
          return 0;
        }
        return 1;
      }
    }
LABEL_2:
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  v29 = 0;
  v23 = *(_BYTE *)(a1 + 120) == 0;
  LOBYTE(v29) = 1;
  v30 = 0;
  v31 = 1;
  if ( v23 || *(_QWORD *)(a1 + 104) || *(_BYTE *)(a1 + 112) )
  {
    v24 = _mm_loadu_si128(&v30);
    *(_QWORD *)(a1 + 120) = v31;
    *(__m128i *)(a1 + 104) = v24;
    return 0;
  }
  return 1;
}
