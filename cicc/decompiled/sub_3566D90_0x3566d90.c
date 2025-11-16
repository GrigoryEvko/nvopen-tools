// Function: sub_3566D90
// Address: 0x3566d90
//
__int64 __fastcall sub_3566D90(__int64 a1, __int64 a2)
{
  unsigned int v4; // r12d
  __int64 *v5; // rbx
  __int64 *i; // r14
  __int64 v7; // rsi
  __int64 v8; // rdi

  v4 = 0;
  v5 = *(__int64 **)(a2 + 8);
  for ( i = *(__int64 **)(a2 + 16); i != v5; v4 |= sub_3566D90(a1, v7) )
    v7 = *v5++;
  sub_3542800(a1, a2);
  if ( !(unsigned __int8)sub_3551890(a1, a2) )
  {
    sub_35492B0(*(__int64 ***)(a1 + 208), a2);
    v8 = *(_QWORD *)(a1 + 784);
    *(_QWORD *)(a1 + 784) = 0;
    if ( !v8 )
      return v4;
    goto LABEL_7;
  }
  if ( !sub_3542A70() )
  {
    if ( !(unsigned __int8)sub_3542A80(a1, (unsigned __int8)v4) )
      goto LABEL_6;
    goto LABEL_9;
  }
  v4 = sub_3566450(a1, a2);
  if ( (unsigned __int8)sub_3542A80(a1, (unsigned __int8)v4) )
LABEL_9:
    v4 = sub_3566B40((_QWORD *)a1, a2);
LABEL_6:
  v8 = *(_QWORD *)(a1 + 784);
  *(_QWORD *)(a1 + 784) = 0;
  if ( v8 )
LABEL_7:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  return v4;
}
