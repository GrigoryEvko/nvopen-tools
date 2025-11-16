// Function: sub_70FDD0
// Address: 0x70fdd0
//
__int64 __fastcall sub_70FDD0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  int v5; // r15d
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rax
  char v9; // r15
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v18[25]; // [rsp+10h] [rbp-190h] BYREF

  v5 = sub_8D32E0(a3);
  v6 = sub_73A720(a1);
  v7 = v5 == 0 ? 5 : 7;
  v8 = sub_72EC50(v6);
  v9 = v5 == 0 ? 5 : 7;
  if ( (*(_BYTE *)(v8 - 8) & 1) != 0 )
    v6 = v8;
  if ( (*(_BYTE *)(v6 + 27) & 2) != 0 && *(_BYTE *)(v6 + 24) == 1 && *(_BYTE *)(v6 + 56) == v9 )
    v6 = *(_QWORD *)(v6 + 72);
  v10 = sub_73DBF0(v7, a3, v6);
  v11 = (__int64 *)v10;
  if ( a4 )
  {
    if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
    {
      sub_6E70E0((__int64 *)v10, (__int64)v18);
      v11 = (__int64 *)sub_6F6F40(v18, 0, v13, v14, v15, v16);
    }
  }
  else
  {
    *(_BYTE *)(v10 + 27) |= 2u;
  }
  result = sub_70FD90(v11, a2);
  *(_BYTE *)(a2 + 177) |= 0x20u;
  return result;
}
