// Function: sub_38606B0
// Address: 0x38606b0
//
bool __fastcall sub_38606B0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v4; // r14
  __int64 v5; // rax

  v2 = *(_QWORD *)(*a1 + 112LL);
  if ( !sub_1456C80(v2, *a2) )
    return 0;
  v4 = a1[3];
  v5 = sub_146F1B0(v2, (__int64)a2);
  return sub_146CEE0(v2, v5, v4);
}
