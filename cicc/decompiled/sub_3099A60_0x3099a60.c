// Function: sub_3099A60
// Address: 0x3099a60
//
__int64 __fastcall sub_3099A60(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  const char *v4; // r14
  size_t v5; // rdx
  size_t v6; // r13
  __int64 v7; // r12
  int v8; // eax
  int v9; // eax
  unsigned int v10; // r8d
  __int64 v11; // rax

  if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
    return 1;
  v3 = *a1;
  v4 = sub_BD5D20(a2);
  v6 = v5;
  v7 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  v8 = sub_C92610();
  v9 = sub_C92860((__int64 *)v3, v4, v6, v8);
  if ( v9 == -1 )
    v11 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  else
    v11 = *(_QWORD *)v3 + 8LL * v9;
  LOBYTE(v10) = v7 == v11;
  return v10;
}
