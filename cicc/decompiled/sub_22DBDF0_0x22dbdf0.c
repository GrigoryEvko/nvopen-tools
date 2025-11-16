// Function: sub_22DBDF0
// Address: 0x22dbdf0
//
bool __fastcall sub_22DBDF0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r8d
  bool result; // al

  v3 = (_QWORD *)(a2 + 48);
  v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v5 == v3 )
    goto LABEL_6;
  if ( !v5 )
    BUG();
  v6 = v5 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
LABEL_6:
    v6 = 0;
    return a3 == sub_B46EC0(v6, 0);
  }
  v7 = sub_B46E30(v6);
  result = 0;
  if ( v7 <= 1 )
    return a3 == sub_B46EC0(v6, 0);
  return result;
}
