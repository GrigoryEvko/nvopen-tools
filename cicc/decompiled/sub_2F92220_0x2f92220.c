// Function: sub_2F92220
// Address: 0x2f92220
//
__int64 __fastcall sub_2F92220(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  char v7; // r8
  __int64 v8; // rdi
  int v9; // r9d
  unsigned int v10; // r10d
  __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 *v15; // rbx
  __int64 v16; // r12
  int i; // r15d
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // eax
  int v21; // r11d

  v7 = *(_BYTE *)(a3 + 8) & 1;
  if ( v7 )
  {
    v8 = a3 + 16;
    v9 = 3;
  }
  else
  {
    v18 = *(unsigned int *)(a3 + 24);
    v8 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v18 )
      goto LABEL_14;
    v9 = v18 - 1;
  }
  v10 = v9 & (37 * a4);
  result = v8 + 16LL * v10;
  v12 = *(_QWORD *)result;
  if ( a4 == *(_QWORD *)result )
    goto LABEL_4;
  v20 = 1;
  while ( v12 != -4096 )
  {
    v21 = v20 + 1;
    v10 = v9 & (v20 + v10);
    result = v8 + 16LL * v10;
    v12 = *(_QWORD *)result;
    if ( a4 == *(_QWORD *)result )
      goto LABEL_4;
    v20 = v21;
  }
  if ( v7 )
  {
    v19 = 64;
    goto LABEL_15;
  }
  v18 = *(unsigned int *)(a3 + 24);
LABEL_14:
  v19 = 16 * v18;
LABEL_15:
  result = v8 + v19;
LABEL_4:
  v13 = 64;
  if ( !v7 )
    v13 = 16LL * *(unsigned int *)(a3 + 24);
  if ( result != v8 + v13 )
  {
    v14 = *(_QWORD *)(a3 + 80) + 32LL * *(unsigned int *)(result + 8);
    result = *(_QWORD *)(a3 + 80) + 32LL * *(unsigned int *)(a3 + 88);
    if ( v14 != result )
    {
      v15 = *(__int64 **)(v14 + 8);
      v16 = v14 + 8;
      for ( i = *(_DWORD *)(a3 + 228); (__int64 *)v16 != v15; v15 = (__int64 *)*v15 )
        result = sub_2F920F0(a1, a2, (__int64 *)v15[2], i);
    }
  }
  return result;
}
