// Function: sub_68B9A0
// Address: 0x68b9a0
//
__int64 __fastcall sub_68B9A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v6; // r12
  __int64 v7; // r13

  v1 = a1;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a1) )
    sub_8AE000(a1);
  v2 = 0;
  v3 = sub_8D4130(a1);
  if ( (unsigned int)sub_8D23B0(v3) )
    return v2;
  if ( !(unsigned int)sub_8D3070(a1) )
  {
    if ( (unsigned int)sub_8D3110(a1) )
      v1 = sub_8D46C0(a1);
    v2 = sub_6E2F40(0);
    v4 = *(_QWORD *)(v2 + 24) + 8LL;
    sub_6EA0A0(v1, v4);
    if ( !(unsigned int)sub_8D2310(v1) )
      sub_6ED1A0(v4);
    return v2;
  }
  v6 = sub_8D46C0(a1);
  v7 = sub_6E2F40(0);
  sub_6EA0A0(v6, *(_QWORD *)(v7 + 24) + 8LL);
  return v7;
}
