// Function: sub_720BA0
// Address: 0x720ba0
//
__int64 __fastcall sub_720BA0(__int64 a1)
{
  __int64 result; // rax
  bool v2; // zf
  __int64 v3; // rdx

  result = qword_4F07940;
  if ( qword_4F07940 )
    qword_4F07940 = *(_QWORD *)(qword_4F07940 + 16);
  else
    result = sub_822B10(24);
  *(_QWORD *)result = a1;
  *(_DWORD *)(result + 8) = 0;
  v2 = unk_4F07678 == 0;
  *(_QWORD *)(result + 16) = 0;
  if ( v2 )
  {
    unk_4F07678 = result;
    qword_4F07938 = result;
  }
  else
  {
    v3 = qword_4F07938;
    qword_4F07938 = result;
    *(_QWORD *)(v3 + 16) = result;
  }
  return result;
}
