// Function: sub_15197A0
// Address: 0x15197a0
//
__int64 __fastcall sub_15197A0(__int64 a1, unsigned int a2)
{
  _QWORD *v3; // rax
  __int64 v4; // r14

  if ( a2 < *(_DWORD *)(a1 + 8) && *(_QWORD *)(*(_QWORD *)a1 + 8LL * a2) )
    return *(_QWORD *)(*(_QWORD *)a1 + 8LL * a2);
  v3 = (_QWORD *)(*(_QWORD *)(a1 + 632) + 16LL * a2);
  v4 = sub_161FF10(*(_QWORD *)(a1 + 240), *v3, v3[1]);
  sub_15194E0(a1, v4, a2);
  return v4;
}
