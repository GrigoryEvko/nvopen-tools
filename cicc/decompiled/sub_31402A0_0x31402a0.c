// Function: sub_31402A0
// Address: 0x31402a0
//
__int64 __fastcall sub_31402A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_B2D7E0(a1, "llvm.assume", qword_49D8C18);
  v3 = 0;
  v5 = v2;
  if ( v2 )
    return (unsigned int)sub_313FFF0(&v5, a2);
  return v3;
}
