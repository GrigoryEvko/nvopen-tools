// Function: sub_131A720
// Address: 0x131a720
//
__int64 __fastcall sub_131A720(_BYTE *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  unsigned int v3; // r13d
  int v5; // edx
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r14

  v1 = qword_5260DD8;
  v3 = sub_13196F0(a1, qword_5260DD8);
  if ( (_BYTE)v3 )
    return v3;
  v5 = sub_1300B70(a1, v1, v2);
  if ( !v5 )
    return v3;
  v6 = qword_50579C0;
  v7 = &qword_50579C0[1];
  v8 = &qword_50579C0[(unsigned int)(v5 - 1) + 1];
  while ( 1 )
  {
    if ( *v6 )
      sub_130B8C0((__int64)a1, *v6 + 10648LL, 0);
    v6 = v7;
    if ( v7 == v8 )
      break;
    ++v7;
  }
  return v3;
}
