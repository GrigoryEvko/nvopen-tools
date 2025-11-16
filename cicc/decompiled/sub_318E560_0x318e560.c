// Function: sub_318E560
// Address: 0x318e560
//
_QWORD *__fastcall sub_318E560(__int64 *a1)
{
  __int64 v1; // r12
  unsigned int v2; // esi
  int v3; // r11d
  __int64 v4; // rcx
  _QWORD *v5; // r14
  unsigned int v6; // edx
  _QWORD *v7; // rax
  __int64 v8; // r9
  _QWORD *result; // rax
  int v10; // eax
  int v11; // edx
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi

  v13 = *a1;
  v14 = a1[1];
  if ( !*(_QWORD *)(v13 + 24) )
    JUMPOUT(0x318D0D8);
  v1 = *(_QWORD *)(v13 + 24);
  v2 = *(_DWORD *)(v14 + 176);
  if ( !v2 )
    JUMPOUT(0x318D0E0);
  v3 = 1;
  v4 = *(_QWORD *)(v14 + 160);
  v5 = 0;
  v6 = (v2 - 1) & (((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4));
  v7 = (_QWORD *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( v1 == *v7 )
    return (_QWORD *)v7[1];
  while ( v8 != -4096 )
  {
    if ( v8 == -8192 && !v5 )
      v5 = v7;
    v6 = (v2 - 1) & (v3 + v6);
    v7 = (_QWORD *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( v1 == *v7 )
      return (_QWORD *)v7[1];
    ++v3;
  }
  if ( !v5 )
    v5 = v7;
  v10 = *(_DWORD *)(v14 + 168);
  ++*(_QWORD *)(v14 + 152);
  v11 = v10 + 1;
  if ( 4 * (v10 + 1) >= 3 * v2 )
    JUMPOUT(0x318D0E8);
  if ( v2 - *(_DWORD *)(v14 + 172) - v11 <= v2 >> 3 )
    JUMPOUT(0x318D160);
  *(_DWORD *)(v14 + 168) = v11;
  if ( *v5 != -4096 )
    --*(_DWORD *)(v14 + 172);
  *v5 = v1;
  v5[1] = 0;
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
  {
    *result = v1;
    result[1] = v14;
  }
  v12 = v5[1];
  v5[1] = result;
  if ( v12 )
  {
    j_j___libc_free_0(v12);
    return (_QWORD *)v5[1];
  }
  return result;
}
