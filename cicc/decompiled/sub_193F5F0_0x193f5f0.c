// Function: sub_193F5F0
// Address: 0x193f5f0
//
__int64 __fastcall sub_193F5F0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // r14d
  unsigned __int8 v4; // al
  __int64 *v7; // r12
  char v9; // al
  __int64 v10; // rax
  __int64 *v11; // r15
  char v12; // dl
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 *v15; // rsi
  unsigned int v16; // edi
  __int64 *v17; // rcx
  unsigned int v18; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 <= 0x10u )
  {
    LOBYTE(v3) = v4 != 9;
    return v3;
  }
  LOBYTE(v3) = v4 > 0x17u && a3 != 6;
  if ( !(_BYTE)v3 )
    return 0;
  v7 = (__int64 *)a1;
  if ( (unsigned __int8)sub_15F2ED0(a1) )
    return 0;
  v9 = *(_BYTE *)(a1 + 16);
  if ( v9 == 29 || v9 == 78 )
    return 0;
  v10 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v11 = *(__int64 **)(a1 - 8);
    v7 = &v11[v10];
  }
  else
  {
    v11 = (__int64 *)(a1 - v10 * 8);
  }
  v18 = a3 + 1;
  while ( v7 != v11 )
  {
    v13 = *v11;
    v14 = *(__int64 **)(a2 + 8);
    if ( *(__int64 **)(a2 + 16) == v14 )
    {
      v15 = &v14[*(unsigned int *)(a2 + 28)];
      v16 = *(_DWORD *)(a2 + 28);
      if ( v14 != v15 )
      {
        v17 = 0;
        while ( v13 != *v14 )
        {
          if ( *v14 == -2 )
            v17 = v14;
          if ( v15 == ++v14 )
          {
            if ( !v17 )
              goto LABEL_26;
            *v17 = v13;
            --*(_DWORD *)(a2 + 32);
            ++*(_QWORD *)a2;
            goto LABEL_14;
          }
        }
        goto LABEL_15;
      }
LABEL_26:
      if ( v16 < *(_DWORD *)(a2 + 24) )
      {
        *(_DWORD *)(a2 + 28) = v16 + 1;
        *v15 = v13;
        ++*(_QWORD *)a2;
LABEL_14:
        if ( !(unsigned __int8)sub_193F5F0(v13, a2, v18) )
          return 0;
        goto LABEL_15;
      }
    }
    sub_16CCBA0(a2, *v11);
    if ( v12 )
      goto LABEL_14;
LABEL_15:
    v11 += 3;
  }
  return v3;
}
