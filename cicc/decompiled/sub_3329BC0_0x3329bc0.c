// Function: sub_3329BC0
// Address: 0x3329bc0
//
__int64 __fastcall sub_3329BC0(unsigned __int16 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  char v4; // dl
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  char v15; // [rsp+18h] [rbp-28h]

  if ( a1 <= 1u
    || (unsigned __int16)(a1 - 504) <= 7u
    || (v2 = a2,
        v3 = 16LL * (a1 - 1),
        v4 = byte_444C4A0[v3 + 8],
        v5 = *(_QWORD *)&byte_444C4A0[v3],
        v15 = v4,
        v14 = v5,
        v6 = sub_CA1930(&v14),
        (unsigned __int16)a2 <= 1u)
    || (unsigned __int16)(a2 - 504) <= 7u )
  {
    BUG();
  }
  v7 = 16LL * ((unsigned __int16)a2 - 1);
  v8 = byte_444C4A0[v7 + 8];
  v9 = *(_QWORD *)&byte_444C4A0[v7];
  v13 = v8;
  v12 = v9;
  v10 = v6 / sub_CA1930(&v12);
  if ( (_DWORD)v10 != 1 )
    return (unsigned int)sub_2D43050(a2, v10);
  return v2;
}
