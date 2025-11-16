// Function: sub_71AD70
// Address: 0x71ad70
//
__int64 __fastcall sub_71AD70(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  if ( (unsigned int)sub_8D32E0(*(_QWORD *)(a1 + 120)) )
  {
    v1 = sub_73E830(a1);
    *(_QWORD *)(v1 + 28) = *(_QWORD *)dword_4F07508;
    result = sub_73DDB0(v1);
  }
  else
  {
    result = sub_731250(a1);
  }
  *(_QWORD *)(result + 28) = *(_QWORD *)dword_4F07508;
  return result;
}
