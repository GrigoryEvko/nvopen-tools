// Function: sub_FFBE30
// Address: 0xffbe30
//
_QWORD *__fastcall sub_FFBE30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 i; // r14
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r13
  _QWORD *result; // rax
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]

  v2 = *(_QWORD *)(a2 + 48);
  for ( i = v2 & 0xFFFFFFFFFFFFFFF8LL; (v2 & 0xFFFFFFFFFFFFFFF8LL) != a2 + 48; i = v2 & 0xFFFFFFFFFFFFFFF8LL )
  {
    if ( !i )
      BUG();
    if ( *(_QWORD *)(i - 8) )
    {
      v4 = sub_ACADE0(*(__int64 ***)(i - 16));
      sub_BD84D0(i - 24, v4);
      v2 = *(_QWORD *)(a2 + 48);
    }
    v5 = v2 & 0xFFFFFFFFFFFFFFF8LL;
    v6 = (_QWORD *)(v5 - 24);
    if ( !v5 )
      v6 = 0;
    sub_B43D60(v6);
    v2 = *(_QWORD *)(a2 + 48);
  }
  v7 = sub_AA48A0(a2);
  sub_B43C20((__int64)&v9, a2);
  result = sub_BD2C40(72, unk_3F148B8);
  if ( result )
    return sub_B4C8A0((__int64)result, v7, v9, v10);
  return result;
}
