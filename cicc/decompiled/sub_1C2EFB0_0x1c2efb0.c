// Function: sub_1C2EFB0
// Address: 0x1c2efb0
//
char __fastcall sub_1C2EFB0(__int64 a1, char a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 *v6; // rax
  _QWORD *v7; // rax
  __int64 v9; // [rsp+8h] [rbp-18h] BYREF

  v3 = (_QWORD *)(a1 + 112);
  if ( a2 )
  {
    LOBYTE(v4) = sub_15602E0(v3, "nvvm.kernel", 0xBu);
    if ( !(_BYTE)v4 )
    {
      v6 = (__int64 *)sub_15E0530(a1);
      v7 = sub_155D020(v6, "nvvm.kernel", 0xBu, 0, 0);
      LOBYTE(v4) = sub_15E0DA0(a1, -1, (__int64)v7);
    }
  }
  else
  {
    LOBYTE(v4) = sub_15602E0(v3, "nvvm.kernel", 0xBu);
    if ( (_BYTE)v4 )
    {
      v9 = *(_QWORD *)(a1 + 112);
      v5 = (__int64 *)sub_15E0530(a1);
      v4 = sub_1563170(&v9, v5, -1, "nvvm.kernel", 0xBu);
      *(_QWORD *)(a1 + 112) = v4;
    }
  }
  return v4;
}
