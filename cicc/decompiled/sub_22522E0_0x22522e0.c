// Function: sub_22522E0
// Address: 0x22522e0
//
__int64 __fastcall sub_22522E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const char *v5; // rdi
  const char *v7; // rsi
  unsigned int v8; // r12d

  v5 = *(const char **)(a1 + 8);
  v7 = *(const char **)(a2 + 8);
  if ( v5 == v7 || (v8 = 0, *v5 != 42) && !strcmp(v5, v7) )
  {
    *(_QWORD *)a4 = a3;
    v8 = 1;
    *(_QWORD *)(a4 + 16) = 16;
    *(_DWORD *)(a4 + 8) = 6;
  }
  return v8;
}
