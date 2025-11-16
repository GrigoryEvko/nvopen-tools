// Function: sub_F20750
// Address: 0xf20750
//
__int64 __fastcall sub_F20750(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v6[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 <= 0x1Cu || *(_QWORD *)(v3 + 40) == **(_QWORD **)a1 )
    return 0;
  v6[1] = v2;
  v5 = *(_QWORD *)(a1 + 8);
  v6[0] = v3;
  sub_F200C0(*(_QWORD *)(v5 + 40) + 2096LL, v6);
  return 1;
}
