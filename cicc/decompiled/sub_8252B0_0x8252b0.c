// Function: sub_8252B0
// Address: 0x8252b0
//
void __fastcall sub_8252B0(__int64 a1, char a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rdi

  if ( a2 == 13 )
  {
    if ( (*(_BYTE *)(qword_4F5F760 + 198) & 0x10) != 0 && *(_BYTE *)(a1 + 24) == 3 )
    {
      v10 = *(_BYTE **)(a1 + 56);
      if ( (v10[156] & 1) == 0 && (v10[174] & 4) != 0 && v10[136] == 2 )
        sub_7E1230(v10, 0, 0, 0);
    }
    v7 = sub_72B0F0(a1, 0);
    if ( v7 )
    {
      v8 = sub_822B10(24, 0, v3, v4, v5, v6);
      v9 = qword_4F5F768;
      *(_QWORD *)v8 = v7;
      *(_QWORD *)(v8 + 16) = v9;
      *(_DWORD *)(v8 + 8) = 0;
      qword_4F5F768 = v8;
    }
  }
}
