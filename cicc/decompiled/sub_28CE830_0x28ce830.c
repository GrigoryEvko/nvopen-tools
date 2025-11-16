// Function: sub_28CE830
// Address: 0x28ce830
//
__int64 __fastcall sub_28CE830(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v8; // r14
  int v9; // r13d
  int v10; // eax
  __int64 v11; // rsi
  int v12; // r8d
  __int64 *v13; // rdi
  unsigned int i; // edx
  __int64 *v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // edx
  unsigned int v18; // [rsp+8h] [rbp-28h] BYREF
  unsigned int v19[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v4 - 1;
  v18 = ((unsigned int)a2[1] >> 4) ^ ((unsigned int)a2[1] >> 9);
  v19[0] = ((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9);
  v10 = sub_28052C0(v19, &v18);
  v11 = *a2;
  v12 = 1;
  v13 = 0;
  for ( i = v9 & v10; ; i = v9 & v17 )
  {
    v15 = (__int64 *)(v8 + 16LL * i);
    v16 = *v15;
    if ( v11 == *v15 && a2[1] == v15[1] )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -8192 && !v13 )
      v13 = (__int64 *)(v8 + 16LL * i);
LABEL_10:
    v17 = v12 + i;
    ++v12;
  }
  if ( v15[1] != -4096 )
    goto LABEL_10;
  if ( !v13 )
    v13 = (__int64 *)(v8 + 16LL * i);
  *a3 = v13;
  return 0;
}
