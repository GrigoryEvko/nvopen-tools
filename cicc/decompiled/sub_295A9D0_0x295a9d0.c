// Function: sub_295A9D0
// Address: 0x295a9d0
//
__int64 __fastcall sub_295A9D0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r8
  int v6; // edx
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // r9
  __int64 result; // rax
  __int64 v11; // rcx
  int v12; // eax
  int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // r13
  __int64 v17; // rcx
  const char *v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  int v22; // edi
  __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a1;
  v5 = **(_QWORD **)(a2 + 32);
  v6 = *(_DWORD *)(*a1 + 16);
  v23[0] = v5;
  if ( !v6 )
  {
    v7 = *(_QWORD **)(v4 + 32);
    v8 = (__int64)&v7[*(unsigned int *)(v4 + 40)];
    v9 = sub_2957650(v7, v8, v23);
    result = 0;
    if ( (_QWORD *)v8 == v9 )
      return result;
    goto LABEL_6;
  }
  v11 = *(_QWORD *)(v4 + 8);
  v12 = *(_DWORD *)(v4 + 24);
  if ( !v12 )
    return 0;
  v13 = v12 - 1;
  v14 = (v12 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v15 = *(_QWORD *)(v11 + 8LL * v14);
  if ( v5 != v15 )
  {
    v22 = 1;
    while ( v15 != -4096 )
    {
      v14 = v13 & (v22 + v14);
      v15 = *(_QWORD *)(v11 + 8LL * v14);
      if ( v5 == v15 )
        goto LABEL_6;
      ++v22;
    }
    return 0;
  }
LABEL_6:
  v16 = a1[1];
  v17 = 14;
  v18 = "<unnamed loop>";
  if ( v5 && (*(_BYTE *)(v5 + 7) & 0x10) != 0 )
  {
    v18 = sub_BD5D20(v5);
    v17 = v21;
  }
  v19 = a2;
  sub_22D0060(*(_QWORD *)(v16 + 8), a2, (__int64)v18, v17);
  if ( a2 == *(_QWORD *)(v16 + 16) )
    *(_BYTE *)(v16 + 24) = 1;
  v20 = *(_QWORD *)a1[2];
  if ( v20 )
  {
    v19 = 0;
    sub_D9D700(v20, 0);
  }
  sub_D47BB0(a2, v19);
  return 1;
}
