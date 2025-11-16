// Function: sub_10E0490
// Address: 0x10e0490
//
__int64 *__fastcall sub_10E0490(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax

  v3 = 0;
  v4 = a3;
  if ( *(char *)(a2 + 7) < 0 )
    v3 = sub_BD2BC0(a2);
  v5 = v3 + 16 * v4;
  v6 = 32LL * *(unsigned int *)(v5 + 8);
  v7 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  a1[2] = *(_QWORD *)v5;
  v8 = v6 - v7 + a2;
  v9 = 32LL * *(unsigned int *)(v5 + 12);
  *a1 = v8;
  a1[1] = (v9 - v6) >> 5;
  return a1;
}
