// Function: sub_210DE90
// Address: 0x210de90
//
__int64 __fastcall sub_210DE90(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD v6[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( (**(_BYTE **)(a2 + 352) & 0x10) != 0 )
  {
    if ( *(_BYTE *)(a1 + 233) )
      sub_16BD130("Instruction selection failed", 1u);
    sub_1E0FED0(a2);
    sub_1E10FF0(a2);
    v2 = *(unsigned __int8 *)(a1 + 232);
    if ( (_BYTE)v2 )
    {
      v4 = *(_QWORD *)a2;
      v6[1] = 0x100000006LL;
      v6[2] = v4;
      v6[0] = &unk_49ED058;
      v5 = sub_15E0530(v4);
      sub_16027F0(v5, (__int64)v6);
    }
    else
    {
      v2 = 1;
    }
  }
  else
  {
    v2 = 0;
  }
  sub_1E69980(*(_QWORD *)(a2 + 40));
  return v2;
}
