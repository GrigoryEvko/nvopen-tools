// Function: sub_31402F0
// Address: 0x31402f0
//
__int64 __fastcall sub_31402F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a1 - 32);
  if ( !v3
    || *(_BYTE *)v3
    || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 80)
    || (result = sub_31402A0(v3, a2), !(_BYTE)result) )
  {
    v6[0] = *(_QWORD *)(a1 + 72);
    v4 = sub_A747B0(v6, -1, "llvm.assume", qword_49D8C18);
    if ( v4 )
    {
      v6[0] = v4;
    }
    else
    {
      v6[0] = sub_B49600(a1, "llvm.assume", qword_49D8C18);
      if ( !v6[0] )
        return 0;
    }
    return sub_313FFF0(v6, a2);
  }
  return result;
}
