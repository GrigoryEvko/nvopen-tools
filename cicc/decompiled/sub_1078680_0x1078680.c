// Function: sub_1078680
// Address: 0x1078680
//
__int64 __fastcall sub_1078680(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v5; // rax
  __int64 v6; // r8
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // edx
  int v14; // r10d
  _QWORD v15[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v16; // [rsp-18h] [rbp-18h]
  __int64 v17; // [rsp-8h] [rbp-8h]

  if ( a3 != 6 )
    return *(unsigned int *)(a2 + 16);
  v5 = *(unsigned int *)(a1 + 192);
  v6 = *(_QWORD *)(a1 + 176);
  if ( !(_DWORD)v5 )
  {
LABEL_7:
    v17 = v3;
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v10 = *(__int64 **)(a2 - 8);
      v11 = *v10;
      v12 = v10 + 3;
    }
    else
    {
      v11 = 0;
      v12 = 0;
    }
    v16 = 1283;
    v15[0] = "symbol not found in type index space: ";
    v15[2] = v12;
    v15[3] = v11;
    sub_C64D30((__int64)v15, 1u);
  }
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v13 = 1;
    while ( v9 != -4096 )
    {
      v14 = v13 + 1;
      v7 = (v5 - 1) & (v13 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_5;
      v13 = v14;
    }
    goto LABEL_7;
  }
LABEL_5:
  if ( v8 == (__int64 *)(v6 + 16 * v5) )
    goto LABEL_7;
  return *((unsigned int *)v8 + 2);
}
