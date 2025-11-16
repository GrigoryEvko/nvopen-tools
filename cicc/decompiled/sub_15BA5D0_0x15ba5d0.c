// Function: sub_15BA5D0
// Address: 0x15ba5d0
//
__int64 __fastcall sub_15BA5D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 *v15; // r8
  __int64 v16; // rcx
  int v17; // r9d
  __int64 *v18; // r10
  int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-68h] BYREF
  int v21; // [rsp+1Ch] [rbp-54h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-50h] BYREF
  int v23; // [rsp+40h] [rbp-30h]
  int v24; // [rsp+44h] [rbp-2Ch] BYREF
  __int64 v25[5]; // [rsp+48h] [rbp-28h] BYREF

  v20 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v20;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v22[0] = 0;
  v10 = *(_QWORD *)(a3 + 8);
  v22[1] = 0;
  v11 = v4 - 1;
  v12 = (-8 * (1LL - *(unsigned int *)(a1 + 8))) >> 3;
  v22[2] = a1 + 8 * (1LL - *(unsigned int *)(a1 + 8));
  v22[3] = v12;
  v23 = *(_DWORD *)(a1 + 4);
  v24 = *(unsigned __int16 *)(a1 + 2);
  v13 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v21 = v23;
  v25[0] = v13;
  v8 = v20;
  v14 = v11 & sub_15B64F0(&v21, &v24, v25);
  v15 = (__int64 *)(v10 + 8LL * v14);
  result = v20;
  v16 = *v15;
  if ( *v15 == v20 )
    return result;
  v17 = 1;
  v7 = 0;
  while ( v16 != -8 )
  {
    if ( v16 != -16 || v7 )
      v15 = v7;
    v14 = v11 & (v17 + v14);
    v18 = (__int64 *)(v10 + 8LL * v14);
    v16 = *v18;
    if ( *v18 == v20 )
      return result;
    ++v17;
    v7 = v15;
    v15 = (__int64 *)(v10 + 8LL * v14);
  }
  v19 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v15;
  ++*(_QWORD *)a3;
  v9 = v19 + 1;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_15BA380(a3, v6);
  sub_15B7230(a3, &v20, v22);
  v7 = (__int64 *)v22[0];
  v8 = v20;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v20;
}
