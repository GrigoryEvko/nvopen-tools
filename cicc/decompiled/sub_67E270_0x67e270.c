// Function: sub_67E270
// Address: 0x67e270
//
__int64 __fastcall sub_67E270(_QWORD *a1, int a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rdx

  v6 = sub_67D720(a1, a2);
  v7 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v7 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_DWORD *)v7 = 2;
  *(_QWORD *)(v7 + 8) = 0;
  *(_QWORD *)(v7 + 16) = unk_4F077C8;
  *(_QWORD *)(v7 + 16) = *a3;
  if ( !*(_QWORD *)(v6 + 184) )
    *(_QWORD *)(v6 + 184) = v7;
  v8 = *(_QWORD *)(v6 + 192);
  if ( v8 )
    *(_QWORD *)(v8 + 8) = v7;
  *(_QWORD *)(v6 + 192) = v7;
  result = sub_67BB20(4);
  *(_QWORD *)(result + 16) = a4;
  *(_DWORD *)(result + 24) = -1;
  if ( !*(_QWORD *)(v6 + 184) )
    *(_QWORD *)(v6 + 184) = result;
  v10 = *(_QWORD *)(v6 + 192);
  if ( v10 )
    *(_QWORD *)(v10 + 8) = result;
  *(_QWORD *)(v6 + 192) = result;
  return result;
}
