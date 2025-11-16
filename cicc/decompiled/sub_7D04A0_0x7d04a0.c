// Function: sub_7D04A0
// Address: 0x7d04a0
//
__int64 __fastcall sub_7D04A0(_QWORD *a1)
{
  _QWORD *v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 result; // rax
  _BYTE v6[96]; // [rsp+0h] [rbp-60h] BYREF

  v2 = (_QWORD *)a1[8];
  sub_878710(a1, v6);
  v3 = sub_7D0130(v2, 3u, 0, (__int64)v6);
  a1[9] = v3;
  v4 = *(_QWORD *)(v3 + 88);
  *(_BYTE *)(v3 + 81) |= 0x80u;
  *(_QWORD *)(v3 + 72) = a1;
  *(_QWORD *)(*(_QWORD *)(v4 + 168) + 8LL) = a1[11];
  result = dword_4F07590;
  if ( dword_4F07590 )
    return sub_7365B0(v4, 0);
  return result;
}
