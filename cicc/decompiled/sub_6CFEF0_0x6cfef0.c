// Function: sub_6CFEF0
// Address: 0x6cfef0
//
__int64 __fastcall sub_6CFEF0(unsigned __int16 a1, _QWORD *a2, __int64 *a3, __m128i *a4, __int64 *a5)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // r15
  _QWORD *v9; // rax
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-1B8h]
  _BYTE v17[16]; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v18[50]; // [rsp+30h] [rbp-190h] BYREF

  if ( !a2 )
    BUG();
  v7 = a2;
  v8 = 0;
  while ( 1 )
  {
    v9 = (_QWORD *)*v7;
    *v7 = v8;
    if ( !v9 )
      break;
    v8 = v7;
    v7 = v9;
  }
  *v7 = 0;
  sub_6E6610(v7, a5, 1);
  result = sub_6E18E0(a5);
  if ( v8 )
  {
    v11 = *(_QWORD *)(qword_4D03C50 + 136LL);
    *(_QWORD *)(qword_4D03C50 + 136LL) = v17;
    v14 = v11;
    do
    {
      sub_6E1BE0(v17);
      v12 = sub_6E3060(a5);
      sub_6E1C20(v12, 1, v17);
      v13 = v8;
      v8 = (_QWORD *)*v8;
      *v13 = 0;
      sub_6E6610(v13, v18, 1);
      sub_6E18E0(v18);
      sub_6CFD10(a1, v18, *a3, a4, a5);
      a5[11] = 0;
    }
    while ( v8 );
    result = qword_4D03C50;
    *(_QWORD *)(qword_4D03C50 + 136LL) = v14;
  }
  return result;
}
