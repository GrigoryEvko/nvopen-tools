// Function: sub_2C725F0
// Address: 0x2c725f0
//
__int64 __fastcall sub_2C725F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8

  v4 = sub_BC1CD0(a4, &unk_4F81450, a3);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v4 + 8;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  sub_2C71E10((_QWORD *)a1, (__int64)&unk_4F81450, v5, v6, v7);
  while ( (unsigned __int8)sub_2C70040((__int64 *)a1) )
    ;
  return a1;
}
