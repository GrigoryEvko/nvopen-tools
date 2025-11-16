// Function: sub_164DB80
// Address: 0x164db80
//
__int64 __fastcall sub_164DB80(__int64 a1, int a2, __int64 a3)
{
  int v3; // r15d
  const char *v4; // rbx
  unsigned int v6; // eax
  _QWORD v7[7]; // [rsp+8h] [rbp-38h] BYREF

  LOBYTE(v3) = 53;
  v4 = (const char *)&unk_42ADC24;
  v7[0] = a3;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 16;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  while ( !(unsigned __int8)sub_1560290(v7, a2, v3) )
  {
    if ( v4 == "LLVM IR Parsing" )
      goto LABEL_6;
LABEL_3:
    v3 = *(_DWORD *)v4;
    v4 += 4;
  }
  sub_15606E0((_QWORD *)a1, v3);
  if ( v4 != "LLVM IR Parsing" )
    goto LABEL_3;
LABEL_6:
  if ( (unsigned __int8)sub_1560290(v7, a2, 1) )
  {
    v6 = sub_15603A0(v7, a2);
    sub_1560C00((_QWORD *)a1, v6);
  }
  return a1;
}
