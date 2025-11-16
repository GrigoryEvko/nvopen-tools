// Function: sub_1F6D200
// Address: 0x1f6d200
//
__int64 __fastcall sub_1F6D200(__int64 a1, unsigned int a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rax
  unsigned int v7; // r8d
  __int64 v8; // rdi
  unsigned int v9; // ebx

  v6 = sub_1D1ADA0(a1, a2, a3, a4, a5, a6);
  v7 = 0;
  if ( !v6 )
    return v7;
  v8 = *(_QWORD *)(v6 + 88);
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 > 0x40 )
  {
    LOBYTE(v7) = v9 == (unsigned int)sub_16A57B0(v8 + 24);
    return v7;
  }
  LOBYTE(v7) = *(_QWORD *)(v8 + 24) == 0;
  return v7;
}
