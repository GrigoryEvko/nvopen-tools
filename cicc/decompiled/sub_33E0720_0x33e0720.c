// Function: sub_33E0720
// Address: 0x33e0720
//
__int64 __fastcall sub_33E0720(__int64 a1, unsigned int a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned int v7; // r8d
  __int64 v8; // rdi
  unsigned int v9; // ebx

  v6 = sub_33DFBC0(a1, a2, a3, 1u, a5, a6);
  v7 = 0;
  if ( !v6 )
    return v7;
  v8 = *(_QWORD *)(v6 + 96);
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 > 0x40 )
  {
    LOBYTE(v7) = v9 == (unsigned int)sub_C444A0(v8 + 24);
    return v7;
  }
  LOBYTE(v7) = *(_QWORD *)(v8 + 24) == 0;
  return v7;
}
