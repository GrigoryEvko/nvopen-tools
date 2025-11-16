// Function: sub_AEAAD0
// Address: 0xaeaad0
//
__int64 __fastcall sub_AEAAD0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r12d
  __int64 v7; // r14
  int v8; // r12d
  int v9; // eax
  __int64 v10; // rsi
  int v11; // r8d
  __int64 *v12; // rdi
  unsigned int i; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // edx

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v4 - 1;
  v9 = sub_AEA4A0(a2, a2 + 1);
  v10 = *a2;
  v11 = 1;
  v12 = 0;
  for ( i = v8 & v9; ; i = v8 & v16 )
  {
    v14 = (__int64 *)(v7 + 16LL * i);
    v15 = *v14;
    if ( *v14 == v10 && a2[1] == v14[1] )
    {
      *a3 = v14;
      return 1;
    }
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && v14[1] == -8192 && !v12 )
      v12 = (__int64 *)(v7 + 16LL * i);
LABEL_10:
    v16 = v11 + i;
    ++v11;
  }
  if ( v14[1] != -4096 )
    goto LABEL_10;
  if ( !v12 )
    v12 = (__int64 *)(v7 + 16LL * i);
  *a3 = v12;
  return 0;
}
