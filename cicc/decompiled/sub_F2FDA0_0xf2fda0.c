// Function: sub_F2FDA0
// Address: 0xf2fda0
//
char __fastcall sub_F2FDA0(__int64 a1, _BYTE *a2)
{
  char result; // al
  unsigned __int8 v4; // dl
  const char *v5; // r15
  size_t v6; // rdx
  size_t v7; // r14
  __int64 v8; // r13
  int v9; // eax
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax

  result = sub_B2FC80((__int64)a2);
  if ( result )
    return 1;
  v4 = a2[32] & 0xF;
  if ( v4 == 1 || (a2[33] & 3) == 2 || *a2 == 3 && (a2[80] & 2) != 0 )
    return 1;
  if ( (unsigned int)v4 - 7 <= 1 )
    return result;
  v5 = sub_BD5D20((__int64)a2);
  v7 = v6;
  v8 = *(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 48);
  v9 = sub_C92610();
  v10 = a1 + 40;
  v11 = sub_C92860((__int64 *)(a1 + 40), v5, v7, v9);
  if ( v11 == -1 )
  {
    v12 = *(unsigned int *)(a1 + 48);
    v13 = *(_QWORD *)(a1 + 40) + 8 * v12;
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 40);
    v13 = v12 + 8LL * v11;
  }
  if ( v8 != v13 )
    return 1;
  if ( !*(_QWORD *)(a1 + 24) )
    sub_4263D6(v10, v5, v12);
  return (*(__int64 (__fastcall **)(__int64, _BYTE *))(a1 + 32))(a1 + 8, a2);
}
