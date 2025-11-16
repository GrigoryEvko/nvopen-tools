// Function: sub_2604CE0
// Address: 0x2604ce0
//
__int64 __fastcall sub_2604CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r15
  unsigned int v6; // r13d
  int v7; // edx
  int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 result; // rax
  int v18; // edx
  __int64 v19; // r13
  unsigned int v20; // edi
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rcx
  unsigned int v24; // r9d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // r11d
  int v31; // ebx
  int v32; // eax
  int v33; // r11d
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+20h] [rbp-70h] BYREF
  __int64 v39; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v40[96]; // [rsp+30h] [rbp-60h] BYREF

  v5 = *(unsigned __int8 **)(a2 + 240);
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *v5;
  v37 = *(_QWORD *)(a5 - 32);
  v8 = v7 - 29;
  if ( v7 != 40 )
    goto LABEL_2;
LABEL_14:
  v9 = 32LL * (unsigned int)sub_B491D0((__int64)v5);
  if ( (v5[7] & 0x80u) == 0 )
    goto LABEL_15;
  while ( 1 )
  {
    v10 = sub_BD2BC0((__int64)v5);
    v12 = v10 + v11;
    if ( (v5[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v12 >> 4) )
LABEL_34:
        BUG();
LABEL_15:
      v16 = 0;
      goto LABEL_11;
    }
    if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)v5)) >> 4) )
      goto LABEL_15;
    if ( (v5[7] & 0x80u) == 0 )
      goto LABEL_34;
    v13 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
    if ( (v5[7] & 0x80u) == 0 )
      BUG();
    v14 = sub_BD2BC0((__int64)v5);
    v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
LABEL_11:
    result = (32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF) - 32 - v9 - v16) >> 5;
    if ( v6 >= (unsigned int)result )
      return result;
    v5 = *(unsigned __int8 **)(a2 + 240);
    if ( *(_QWORD *)&v5[32 * (v6 - (unsigned __int64)(*((_DWORD *)v5 + 1) & 0x7FFFFFF))] == v37 )
      break;
    v18 = *v5;
    ++v6;
    v8 = v18 - 29;
    if ( v18 == 40 )
      goto LABEL_14;
LABEL_2:
    v9 = 0;
    if ( v8 != 56 )
    {
      if ( v8 != 5 )
        BUG();
      v9 = 64;
    }
    if ( (v5[7] & 0x80u) == 0 )
      goto LABEL_15;
  }
  v19 = v6 - *(_DWORD *)(a2 + 24);
  v20 = *(_DWORD *)(a1 + 80);
  v21 = *(_QWORD *)(a3 + 8 * v19);
  v22 = a1 + 56;
  v23 = *(_QWORD *)(a1 + 64);
  if ( !v20 )
  {
LABEL_24:
    v39 = *(_QWORD *)(a3 + 8 * v19);
    v38 = a5;
    return sub_2604B50((__int64)v40, v22, &v38, &v39);
  }
  v24 = v20 - 1;
  v25 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v26 = (__int64 *)(v23 + 16LL * v25);
  v27 = *v26;
  if ( v21 != *v26 )
  {
    v29 = *v26;
    v30 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v31 = 1;
    while ( v29 != -4096 )
    {
      v30 = v24 & (v31 + v30);
      v29 = *(_QWORD *)(v23 + 16LL * v30);
      if ( v21 == v29 )
      {
        v32 = 1;
        while ( v27 != -4096 )
        {
          v33 = v32 + 1;
          v25 = v24 & (v32 + v25);
          v26 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( v21 == *v26 )
            goto LABEL_20;
          v32 = v33;
        }
        v26 = (__int64 *)(v23 + 16LL * v20);
        goto LABEL_20;
      }
      ++v31;
    }
    goto LABEL_24;
  }
LABEL_20:
  v28 = v26[1];
  v38 = a5;
  v39 = v28;
  return sub_2604B50((__int64)v40, v22, &v38, &v39);
}
