// Function: sub_190B590
// Address: 0x190b590
//
__int64 __fastcall sub_190B590(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // r12
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  int v10; // [rsp+10h] [rbp-30h]

  v3 = 0;
  v5 = *(_QWORD *)(a1 + 24);
  v9[1] = 0;
  v10 = (int)&loc_1000000;
  v9[0] = v5;
  v6 = sub_157EBA0(a2);
  while ( a3 != sub_15F4DF0(v6, v3) )
    ++v3;
  v7 = sub_1AAC5F0(v6, v3, v9);
  if ( *(_QWORD *)a1 )
    sub_1413520(*(_QWORD *)a1);
  sub_190B570(a1);
  *(_BYTE *)(a1 + 784) = 1;
  return v7;
}
