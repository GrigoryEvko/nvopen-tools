// Function: sub_B9F6F0
// Address: 0xb9f6f0
//
__int64 __fastcall sub_B9F6F0(__int64 *a1, _BYTE *a2)
{
  _BYTE *v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r14d
  _QWORD *v8; // r9
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  _BYTE *v11; // r10
  __int64 result; // rax
  __int64 *v13; // rbx
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // r12
  __int64 v17; // [rsp+0h] [rbp-40h]
  _BYTE *v18; // [rsp+8h] [rbp-38h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-28h] BYREF

  v18 = a2;
  v3 = sub_B9F650(a1, a2);
  v4 = *a1;
  v18 = v3;
  v5 = *(_DWORD *)(v4 + 624);
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 600);
    v19 = 0;
LABEL_22:
    v5 *= 2;
    goto LABEL_23;
  }
  v6 = *(_QWORD *)(v4 + 608);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = (_BYTE *)*v10;
  if ( v3 == (_BYTE *)*v10 )
  {
LABEL_3:
    result = v10[1];
    v13 = v10 + 1;
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != (_BYTE *)-4096LL )
  {
    if ( !v8 && v11 == (_BYTE *)-8192LL )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = (_BYTE *)*v10;
    if ( v3 == (_BYTE *)*v10 )
      goto LABEL_3;
    ++v7;
  }
  v14 = *(_DWORD *)(v4 + 616);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(v4 + 600);
  v15 = v14 + 1;
  v19 = v8;
  if ( 4 * v15 >= 3 * v5 )
    goto LABEL_22;
  if ( v5 - *(_DWORD *)(v4 + 620) - v15 <= v5 >> 3 )
  {
LABEL_23:
    sub_B95C80(v4 + 600, v5);
    sub_B92630(v4 + 600, (__int64 *)&v18, &v19);
    v3 = v18;
    v8 = v19;
    v15 = *(_DWORD *)(v4 + 616) + 1;
  }
  *(_DWORD *)(v4 + 616) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v4 + 620);
  *v8 = v3;
  v13 = v8 + 1;
  v8[1] = 0;
LABEL_18:
  v16 = sub_BCB180(a1);
  result = sub_22077B0(32);
  if ( result )
  {
    v17 = result;
    sub_B96F20(result, v16, (__int64)v18);
    result = v17;
  }
  *v13 = result;
  return result;
}
