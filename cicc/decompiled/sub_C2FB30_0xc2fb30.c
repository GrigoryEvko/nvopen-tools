// Function: sub_C2FB30
// Address: 0xc2fb30
//
unsigned __int64 __fastcall sub_C2FB30(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rcx

  v3 = sub_C935B0(a1, a2, a3, 0);
  v4 = a1[1];
  if ( v3 < v4 )
    v4 = v3;
  return *a1 + v4;
}
