// Function: sub_6018B0
// Address: 0x6018b0
//
__int64 __fastcall sub_6018B0(__int64 a1)
{
  __int64 v2; // rdi
  bool v3; // sf
  __int64 result; // rax
  __int64 v5; // rax

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v3 = *(char *)(v2 + 183) < 0;
  *(_BYTE *)(v2 + 183) &= ~0x40u;
  if ( v3 )
  {
    v5 = sub_5EB340(v2);
    sub_71D150(*(_QWORD *)(v5 + 88));
  }
  result = word_4D04898;
  if ( word_4D04898 )
  {
    sub_600530(a1);
    return sub_6013C0(a1);
  }
  return result;
}
