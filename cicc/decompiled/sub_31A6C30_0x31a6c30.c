// Function: sub_31A6C30
// Address: 0x31a6c30
//
char __fastcall sub_31A6C30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_D47930(*(_QWORD *)a1);
  if ( *(_BYTE *)(a1 + 664) )
    return v2 == a2;
  else
    return sub_D364B0(a2, *(_QWORD *)a1, *(_QWORD *)(a1 + 40));
}
