// Function: sub_1E85D70
// Address: 0x1e85d70
//
__int64 __fastcall sub_1E85D70(__int64 a1, __int64 a2, signed int a3, int a4)
{
  void *v6; // rax
  __int64 v7; // r12
  _BYTE *v8; // rax
  __int64 result; // rax

  v6 = sub_16E8CB0();
  v7 = sub_1263B40((__int64)v6, "- liverange:   ");
  sub_1DB50D0(a2, v7);
  v8 = *(_BYTE **)(v7 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
  {
    sub_16E7DE0(v7, 10);
  }
  else
  {
    *(_QWORD *)(v7 + 24) = v8 + 1;
    *v8 = 10;
  }
  if ( a3 >= 0 )
  {
    result = sub_1E859F0(a1, a3);
    if ( !a4 )
      return result;
    return sub_1E85CD0(a4);
  }
  result = sub_1E85940(a1, a3);
  if ( a4 )
    return sub_1E85CD0(a4);
  return result;
}
