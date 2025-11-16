// Function: sub_318E570
// Address: 0x318e570
//
_QWORD *__fastcall sub_318E570(__int64 a1, int a2)
{
  __int64 v2; // r12
  unsigned int v3; // esi
  int v4; // r11d
  __int64 v5; // rcx
  __int64 *v6; // r14
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  _QWORD *result; // rax
  int v11; // eax
  int v12; // edx
  unsigned __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 savedregs; // [rsp+10h] [rbp+0h] BYREF

  v14 = *(_QWORD *)(a1 + 8);
  v15 = sub_BCDA70(*(__int64 **)a1, a2);
  v16 = v14;
  if ( !v15 )
    JUMPOUT(0x318D0D8);
  savedregs = (__int64)&savedregs;
  v2 = v15;
  v3 = *(_DWORD *)(v16 + 176);
  if ( !v3 )
    JUMPOUT(0x318D0E0);
  v4 = 1;
  v5 = *(_QWORD *)(v16 + 160);
  v6 = 0;
  v7 = (v3 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( v2 == *v8 )
    return (_QWORD *)v8[1];
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v6 )
      v6 = v8;
    v7 = (v3 - 1) & (v4 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( v2 == *v8 )
      return (_QWORD *)v8[1];
    ++v4;
  }
  if ( !v6 )
    v6 = v8;
  v11 = *(_DWORD *)(v16 + 168);
  ++*(_QWORD *)(v16 + 152);
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    JUMPOUT(0x318D0E8);
  if ( v3 - *(_DWORD *)(v16 + 172) - v12 <= v3 >> 3 )
    JUMPOUT(0x318D160);
  *(_DWORD *)(v16 + 168) = v12;
  if ( *v6 != -4096 )
    --*(_DWORD *)(v16 + 172);
  *v6 = v2;
  v6[1] = 0;
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
  {
    *result = v2;
    result[1] = v16;
  }
  v13 = v6[1];
  v6[1] = (__int64)result;
  if ( v13 )
  {
    j_j___libc_free_0(v13);
    return (_QWORD *)v6[1];
  }
  return result;
}
