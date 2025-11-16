// Function: sub_2FF3FD0
// Address: 0x2ff3fd0
//
__int64 __fastcall sub_2FF3FD0(_QWORD *a1)
{
  int *v1; // rax
  int v2; // eax
  _DWORD *v3; // rax
  __int64 v6; // r14
  void (__fastcall *v7)(__int64, __m128i *, __int64); // rbx
  __m128i *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  __int64 (__fastcall *v12)(__int64); // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rsi
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v16)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-30h]

  v1 = (int *)sub_C94E20((__int64)qword_4F86310);
  if ( v1 )
    v2 = *v1;
  else
    v2 = qword_4F86310[2];
  if ( v2 != 4 )
  {
    v3 = sub_C94E20((__int64)qword_4F86310);
    if ( v3 ? *v3 : LODWORD(qword_4F86310[2]) )
      return sub_2FF3D20((__int64)a1);
  }
  if ( sub_23CF310(a1[32]) )
  {
    v13 = (_QWORD *)sub_2E2F870();
    sub_2FF0E80((__int64)a1, v13, 0);
  }
  v6 = a1[22];
  v7 = *(void (__fastcall **)(__int64, __m128i *, __int64))(*(_QWORD *)v6 + 16LL);
  sub_23CF4B0(&v15, a1[32]);
  v8 = sub_DFEEB0(&v15);
  v7(v6, v8, 1);
  if ( v16 )
    v16(&v15, &v15, 3);
  v9 = (_QWORD *)sub_2F3C630();
  sub_2FF0E80((__int64)a1, v9, 0);
  v10 = (_QWORD *)sub_2DBA350();
  sub_2FF0E80((__int64)a1, v10, 0);
  v11 = (_QWORD *)sub_2DBBC40();
  sub_2FF0E80((__int64)a1, v11, 0);
  (*(void (__fastcall **)(_QWORD *))(*a1 + 160LL))(a1);
  v12 = *(__int64 (__fastcall **)(__int64))(*a1 + 168LL);
  if ( v12 == sub_2FF1980 )
  {
    if ( (unsigned int)sub_2FF0570((__int64)a1) )
    {
      if ( !byte_5029108 )
      {
        v14 = (_QWORD *)sub_2D5CDB0();
        sub_2FF0E80((__int64)a1, v14, 0);
      }
    }
  }
  else
  {
    v12((__int64)a1);
  }
  sub_2FF1E40((__int64)a1);
  (*(void (__fastcall **)(_QWORD *))(*a1 + 176LL))(a1);
  return sub_2FF2920(a1);
}
