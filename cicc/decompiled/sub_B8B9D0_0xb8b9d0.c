// Function: sub_B8B9D0
// Address: 0xb8b9d0
//
__int64 __fastcall sub_B8B9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v7; // r12
  char v8; // bl
  __int64 v9; // rax
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v11[0] = a3;
  v7 = *(_QWORD *)sub_B8B720(a2 + 568, v11);
  sub_B808B0(v7);
  v8 = sub_B8A550(v7, a5);
  if ( v7 )
    v7 += 568;
  v9 = sub_B811E0(v7, a4);
  *(_BYTE *)a1 = v8;
  *(_QWORD *)(a1 + 8) = v9;
  return a1;
}
