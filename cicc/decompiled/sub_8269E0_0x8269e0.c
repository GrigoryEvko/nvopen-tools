// Function: sub_8269E0
// Address: 0x8269e0
//
__int64 __fastcall sub_8269E0(unsigned int a1)
{
  __int64 v1; // rax

  if ( !qword_4F1F650 )
    return 0;
  v1 = sub_881B20(qword_4F1F650, a1, 0);
  if ( v1 )
    return *(unsigned int *)(*(_QWORD *)v1 + 4LL);
  else
    return 0;
}
