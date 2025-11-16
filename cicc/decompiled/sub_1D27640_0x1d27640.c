// Function: sub_1D27640
// Address: 0x1d27640
//
__int64 __fastcall sub_1D27640(__int64 a1, char *a2, unsigned __int8 a3, __int64 a4)
{
  size_t v5; // r12
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rcx
  __int64 v11; // r14
  __int64 result; // rax
  __int64 v13; // rax
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  _QWORD *v16; // r14
  unsigned int v17; // eax
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 v20; // r15
  __int128 v21; // rdi
  __int64 v22; // rax
  unsigned __int8 *v23; // rsi
  _QWORD *v26; // [rsp+10h] [rbp-50h]
  _QWORD *v27; // [rsp+10h] [rbp-50h]
  unsigned int v28; // [rsp+1Ch] [rbp-44h]
  unsigned int v29; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 *v30; // [rsp+28h] [rbp-38h] BYREF

  v5 = 0;
  if ( a2 )
    v5 = strlen(a2);
  v8 = (unsigned int)sub_16D19C0(a1 + 792, (unsigned __int8 *)a2, v5);
  v10 = (_QWORD *)(*(_QWORD *)(a1 + 792) + 8 * v8);
  v11 = *v10;
  if ( *v10 )
  {
    if ( v11 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 808);
  }
  v26 = v10;
  v28 = v8;
  v13 = malloc(v5 + 17);
  v14 = v28;
  v15 = v26;
  v16 = (_QWORD *)v13;
  if ( !v13 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v15 = v26;
    v14 = v28;
  }
  if ( v5 )
  {
    v27 = v15;
    v29 = v14;
    memcpy(v16 + 2, a2, v5);
    v15 = v27;
    v14 = v29;
  }
  *((_BYTE *)v16 + v5 + 16) = 0;
  *v16 = v5;
  v16[1] = 0;
  *v15 = v16;
  ++*(_DWORD *)(a1 + 804);
  v17 = sub_16D1CD0(a1 + 792, v14);
  v7 = *(_QWORD *)(a1 + 792);
  v18 = (__int64 *)(v7 + 8LL * v17);
  v11 = *v18;
  if ( *v18 != -8 && v11 )
  {
LABEL_5:
    result = *(_QWORD *)(v11 + 8);
    if ( result )
      return result;
    goto LABEL_18;
  }
  v19 = v18 + 1;
  do
  {
    do
      v11 = *v19++;
    while ( !v11 );
  }
  while ( v11 == -8 );
  result = *(_QWORD *)(v11 + 8);
  if ( !result )
  {
LABEL_18:
    v20 = *(_QWORD *)(a1 + 208);
    if ( v20 )
      *(_QWORD *)(a1 + 208) = *(_QWORD *)v20;
    else
      v20 = sub_145CBF0((__int64 *)(a1 + 216), 112, 8);
    *((_QWORD *)&v21 + 1) = a4;
    *(_QWORD *)&v21 = a3;
    v22 = sub_1D274F0(v21, v7, (__int64)v10, v8, v9);
    v30 = 0;
    *(_QWORD *)v20 = 0;
    v23 = v30;
    *(_QWORD *)(v20 + 40) = v22;
    *(_QWORD *)(v20 + 8) = 0;
    *(_QWORD *)(v20 + 16) = 0;
    *(_WORD *)(v20 + 24) = 17;
    *(_DWORD *)(v20 + 28) = -1;
    *(_QWORD *)(v20 + 32) = 0;
    *(_QWORD *)(v20 + 48) = 0;
    *(_QWORD *)(v20 + 56) = 0x100000000LL;
    *(_DWORD *)(v20 + 64) = 0;
    *(_QWORD *)(v20 + 72) = v23;
    if ( v23 )
      sub_1623210((__int64)&v30, v23, v20 + 72);
    *(_WORD *)(v20 + 80) &= 0xF000u;
    *(_WORD *)(v20 + 26) = 0;
    *(_QWORD *)(v20 + 88) = a2;
    *(_BYTE *)(v20 + 96) = 0;
    *(_QWORD *)(v11 + 8) = v20;
    sub_1D172A0(a1, v20);
    return *(_QWORD *)(v11 + 8);
  }
  return result;
}
