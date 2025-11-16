// Function: sub_B070C0
// Address: 0xb070c0
//
__int64 __fastcall sub_B070C0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  int v10; // eax
  _BYTE *v11; // rdi
  __int64 v12; // r14
  unsigned int v13; // r12d
  unsigned int v14; // edx
  __int64 *v15; // r8
  __int64 v16; // rcx
  int v17; // r9d
  __int64 *v18; // r10
  int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v21; // [rsp+10h] [rbp-30h] BYREF
  __int64 v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v20 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v20;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v21 = 0;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(_DWORD *)(a1 + 20);
  v11 = (_BYTE *)(a1 - 16);
  v12 = *(_QWORD *)(a3 + 8);
  v13 = v4 - 1;
  LODWORD(v21) = v10;
  BYTE4(v21) = v11[60];
  v22[0] = *((_QWORD *)sub_A17150(v11) + 3);
  v8 = v20;
  v14 = v13 & sub_AF8410((int *)&v21, (__int8 *)&v21 + 4, v22);
  v15 = (__int64 *)(v12 + 8LL * v14);
  result = v20;
  v16 = *v15;
  if ( *v15 == v20 )
    return result;
  v17 = 1;
  v7 = 0;
  while ( v16 != -4096 )
  {
    if ( v16 != -8192 || v7 )
      v15 = v7;
    v14 = v13 & (v17 + v14);
    v18 = (__int64 *)(v12 + 8LL * v14);
    v16 = *v18;
    if ( *v18 == v20 )
      return result;
    ++v17;
    v7 = v15;
    v15 = (__int64 *)(v12 + 8LL * v14);
  }
  v19 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v15;
  ++*(_QWORD *)a3;
  v9 = v19 + 1;
  v21 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B06E50(a3, v6);
  sub_AFD910(a3, &v20, &v21);
  v7 = v21;
  v8 = v20;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v20;
}
