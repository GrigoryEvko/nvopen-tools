// Function: sub_67F190
// Address: 0x67f190
//
__int64 __fastcall sub_67F190(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        __int64 a8,
        int a9)
{
  __int64 result; // rax
  __int64 v10; // rdx

  result = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    result = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(result + 8) = 0;
  *(_DWORD *)result = 7;
  *(_BYTE *)(result + 16) = a7;
  *(_QWORD *)(result + 24) = a8;
  *(_DWORD *)(result + 32) = a9;
  if ( !*(_QWORD *)(a1 + 184) )
    *(_QWORD *)(a1 + 184) = result;
  v10 = *(_QWORD *)(a1 + 192);
  if ( v10 )
    *(_QWORD *)(v10 + 8) = result;
  *(_QWORD *)(a1 + 192) = result;
  return result;
}
