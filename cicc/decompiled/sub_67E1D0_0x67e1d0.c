// Function: sub_67E1D0
// Address: 0x67e1d0
//
__int64 __fastcall sub_67E1D0(_QWORD *a1, int a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 v6; // rdx

  v4 = sub_67D720(a1, a2);
  result = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    result = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_DWORD *)result = 4;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 24) = 0xFFFFFFFFLL;
  *(_WORD *)(result + 32) = 0;
  *(_BYTE *)(result + 34) = 0;
  *(_QWORD *)(result + 16) = a3;
  if ( !*(_QWORD *)(v4 + 184) )
    *(_QWORD *)(v4 + 184) = result;
  v6 = *(_QWORD *)(v4 + 192);
  if ( v6 )
    *(_QWORD *)(v6 + 8) = result;
  *(_QWORD *)(v4 + 192) = result;
  return result;
}
