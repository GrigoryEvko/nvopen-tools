// Function: sub_7FA3E0
// Address: 0x7fa3e0
//
_QWORD *__fastcall sub_7FA3E0(_QWORD **a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v5; // r14
  __m128i *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 *v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  _BYTE *v15; // [rsp+8h] [rbp-38h]

  v5 = a1[2];
  v6 = sub_73D4C0((const __m128i *)*a1, dword_4F077C4 == 2);
  a1[2] = 0;
  if ( (*((_BYTE *)a1 + 25) & 1) != 0 )
  {
    v7 = sub_72D2E0(v6);
    v8 = sub_7E7CB0(v7);
    v15 = sub_73E1B0((__int64)a1, 0);
    v9 = sub_72D2E0(*a1);
    v10 = sub_73E130(v15, v9);
  }
  else
  {
    v8 = sub_7E7CB0((__int64)v6);
    v10 = sub_73E130(a1, (__int64)*a1);
  }
  v11 = (__int64 *)sub_7E2BE0(v8, (__int64)v10);
  v12 = sub_73E830(v8);
  *a2 = v12;
  v13 = v12;
  v12[2] = v5;
  if ( *a3 )
    v11 = (__int64 *)sub_73DF90(*a3, v11);
  *a3 = (__int64)v11;
  sub_7304E0((__int64)v11);
  return v13;
}
