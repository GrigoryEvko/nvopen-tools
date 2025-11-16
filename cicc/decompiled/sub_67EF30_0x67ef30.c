// Function: sub_67EF30
// Address: 0x67ef30
//
__int64 __fastcall sub_67EF30(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    result = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_DWORD *)result = 2;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = unk_4F077C8;
  *(_QWORD *)(result + 16) = *a2;
  if ( !*(_QWORD *)(a1 + 184) )
    *(_QWORD *)(a1 + 184) = result;
  v3 = *(_QWORD *)(a1 + 192);
  if ( v3 )
    *(_QWORD *)(v3 + 8) = result;
  *(_QWORD *)(a1 + 192) = result;
  return result;
}
