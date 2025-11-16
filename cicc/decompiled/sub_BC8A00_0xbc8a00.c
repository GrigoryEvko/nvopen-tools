// Function: sub_BC8A00
// Address: 0xbc8a00
//
__int64 __fastcall sub_BC8A00(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  int v3; // ebx

  v1 = sub_BC89C0(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  v3 = sub_BC8980(v1);
  if ( v3 != (unsigned int)sub_B46E30(a1) )
    return 0;
  return v2;
}
