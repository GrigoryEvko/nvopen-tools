// Function: sub_15CB810
// Address: 0x15cb810
//
__int64 __fastcall sub_15CB810(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // rdi
  unsigned __int8 *v4; // rax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned __int8 **)(v3 + 24);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 16) )
  {
    sub_16E7DE0(v3, a2);
  }
  else
  {
    *(_QWORD *)(v3 + 24) = v4 + 1;
    *v4 = a2;
  }
  return a1;
}
