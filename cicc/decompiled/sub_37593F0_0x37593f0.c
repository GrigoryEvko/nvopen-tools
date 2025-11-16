// Function: sub_37593F0
// Address: 0x37593f0
//
unsigned __int64 __fastcall sub_37593F0(__int64 a1, int *a2)
{
  char v3; // dl
  __int64 v4; // r9
  int v5; // ecx
  int v6; // esi
  unsigned int v7; // r8d
  int *v8; // r12
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  int v14; // r10d

  v3 = *(_BYTE *)(a1 + 1536) & 1;
  if ( v3 )
  {
    v4 = a1 + 1544;
    v5 = 7;
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 1552);
    v4 = *(_QWORD *)(a1 + 1544);
    if ( !(_DWORD)v12 )
      goto LABEL_12;
    v5 = v12 - 1;
  }
  v6 = *a2;
  v7 = v5 & (37 * v6);
  v8 = (int *)(v4 + 8LL * v7);
  v9 = *v8;
  if ( v6 == *v8 )
    goto LABEL_4;
  v14 = 1;
  while ( v9 != -1 )
  {
    v7 = v5 & (v14 + v7);
    v8 = (int *)(v4 + 8LL * v7);
    v9 = *v8;
    if ( v6 == *v8 )
      goto LABEL_4;
    ++v14;
  }
  if ( v3 )
  {
    v13 = 64;
    goto LABEL_13;
  }
  v12 = *(unsigned int *)(a1 + 1552);
LABEL_12:
  v13 = 8 * v12;
LABEL_13:
  v8 = (int *)(v4 + v13);
LABEL_4:
  v10 = 64;
  if ( !v3 )
    v10 = 8LL * *(unsigned int *)(a1 + 1552);
  result = v4 + v10;
  if ( v8 != (int *)result )
  {
    sub_37593F0(a1, v8 + 1);
    result = (unsigned int)v8[1];
    *a2 = result;
  }
  return result;
}
