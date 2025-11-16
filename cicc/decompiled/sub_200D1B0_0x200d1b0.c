// Function: sub_200D1B0
// Address: 0x200d1b0
//
unsigned __int64 __fastcall sub_200D1B0(__int64 a1, int *a2)
{
  char v3; // dl
  __int64 v4; // rax
  int v5; // ecx
  int v6; // esi
  int v7; // r10d
  unsigned int v8; // r8d
  int *v9; // r12
  int v10; // r9d
  __int64 v11; // rcx
  unsigned __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx

  v3 = *(_BYTE *)(a1 + 1296) & 1;
  if ( v3 )
  {
    v4 = a1 + 1304;
    v5 = 7;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 1312);
    v4 = *(_QWORD *)(a1 + 1304);
    if ( !(_DWORD)v13 )
      goto LABEL_12;
    v5 = v13 - 1;
  }
  v6 = *a2;
  v7 = 1;
  v8 = v5 & (37 * v6);
  v9 = (int *)(v4 + 8LL * v8);
  v10 = *v9;
  if ( *v9 == v6 )
    goto LABEL_4;
  while ( v10 != -1 )
  {
    v8 = v5 & (v7 + v8);
    v9 = (int *)(v4 + 8LL * v8);
    v10 = *v9;
    if ( v6 == *v9 )
      goto LABEL_4;
    ++v7;
  }
  if ( v3 )
  {
    v14 = 64;
    goto LABEL_13;
  }
  v13 = *(unsigned int *)(a1 + 1312);
LABEL_12:
  v14 = 8 * v13;
LABEL_13:
  v9 = (int *)(v4 + v14);
LABEL_4:
  v11 = 64;
  if ( !v3 )
    v11 = 8LL * *(unsigned int *)(a1 + 1312);
  result = v11 + v4;
  if ( v9 != (int *)result )
  {
    sub_200D1B0(a1, v9 + 1);
    result = (unsigned int)v9[1];
    *a2 = result;
  }
  return result;
}
