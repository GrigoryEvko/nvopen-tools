// Function: sub_13CA540
// Address: 0x13ca540
//
__int64 __fastcall sub_13CA540(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rax

  v4 = sub_13CA510(a1, a2);
  v5 = (_QWORD *)sub_13C9060(v4, a3);
  if ( v5 )
    return sub_13A5BC0(v5, *(_QWORD *)(a1 + 32));
  else
    return 0;
}
