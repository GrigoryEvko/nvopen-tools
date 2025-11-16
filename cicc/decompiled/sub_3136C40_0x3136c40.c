// Function: sub_3136C40
// Address: 0x3136c40
//
unsigned __int64 __fastcall sub_3136C40(
        __int64 a1,
        int a2,
        _BYTE *a3,
        int a4,
        int a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8)
{
  unsigned __int64 result; // rax
  int v12; // r8d
  int v13; // edx
  int v14; // [rsp+4h] [rbp-3Ch]
  int v15; // [rsp+8h] [rbp-38h]

  result = a7;
  v12 = a8;
  if ( *(_BYTE *)(a1 + 338) )
  {
    if ( !*a3 )
    {
      sub_B2CD60((__int64)a3, "kernel", 6u, 0, 0);
      if ( *(_DWORD *)(a1 + 880) == 27 )
        sub_B2CD60((__int64)a3, "uniform-work-group-size", 0x17u, "true", 4u);
      return sub_B2CD30((__int64)a3, 19);
    }
  }
  else
  {
    if ( !a8 )
    {
      v14 = a5;
      v15 = a4;
      LODWORD(result) = (unsigned int)sub_BD5D20((__int64)a3);
      a5 = v14;
      a4 = v15;
      v12 = v13;
    }
    return sub_3717A20(*(_QWORD *)(a1 + 504), 1, a2, result, v12, a4, a5, 0, 0, (__int64)"llvm_offload_entries", 20);
  }
  return result;
}
