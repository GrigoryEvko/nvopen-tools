// Function: sub_643FD0
// Address: 0x643fd0
//
__int64 __fastcall sub_643FD0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = qword_4CFDE68;
  if ( qword_4CFDE68 )
    qword_4CFDE68 = *(_QWORD *)qword_4CFDE68;
  else
    result = sub_823970(56);
  v3 = *(_QWORD *)(a1 + 368);
  *(_BYTE *)(result + 32) &= ~1u;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)result = v3;
  *(_DWORD *)(result + 20) = 0;
  LODWORD(v3) = dword_4F06650[0];
  *(_QWORD *)(result + 24) = a2;
  *(_DWORD *)(result + 16) = v3;
  *(_QWORD *)(result + 36) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(result + 44) = qword_4F063F0;
  *(_QWORD *)(a1 + 368) = result;
  return result;
}
