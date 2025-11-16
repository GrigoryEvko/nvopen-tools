// Function: sub_877320
// Address: 0x877320
//
__int64 __fastcall sub_877320(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rbx
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 i; // r14
  __int64 *v8; // r12
  __m128i v9[24]; // [rsp+0h] [rbp-180h] BYREF

  v2 = a2;
  result = sub_72B840(a1);
  *a2 = 0;
  v4 = *(_QWORD *)(result + 64);
  v5 = result;
  if ( v4 )
  {
    v6 = sub_73E830(v4);
    sub_6E70E0(v6, (__int64)v9);
    result = sub_6E3060(v9);
    *a2 = result;
    v2 = (_QWORD *)result;
  }
  for ( i = *(_QWORD *)(v5 + 40); i; v2 = (_QWORD *)result )
  {
    v8 = sub_73E830(i);
    if ( (unsigned int)sub_8D2FB0(*v8) )
      v8 = (__int64 *)sub_73DDB0(v8);
    sub_6E7170(v8, (__int64)v9);
    result = sub_6E3060(v9);
    *v2 = result;
    i = *(_QWORD *)(i + 112);
  }
  return result;
}
