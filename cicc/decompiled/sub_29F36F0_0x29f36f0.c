// Function: sub_29F36F0
// Address: 0x29f36f0
//
__int64 __fastcall sub_29F36F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v5; // rbx
  unsigned __int8 *v6; // r13
  unsigned __int8 *v7; // rdi
  __int64 i; // r15
  __int64 j; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  const char *v14[4]; // [rsp+10h] [rbp-60h] BYREF
  char v15; // [rsp+30h] [rbp-40h]
  char v16; // [rsp+31h] [rbp-3Fh]

  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, a2, a3, a4);
    v5 = *(unsigned __int8 **)(a3 + 96);
    v6 = &v5[40 * *(_QWORD *)(a3 + 104)];
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a3, a2, v11, v12);
      v5 = *(unsigned __int8 **)(a3 + 96);
    }
  }
  else
  {
    v5 = *(unsigned __int8 **)(a3 + 96);
    v6 = &v5[40 * *(_QWORD *)(a3 + 104)];
  }
  while ( v6 != v5 )
  {
    while ( (v5[7] & 0x10) != 0 )
    {
      v5 += 40;
      if ( v6 == v5 )
        goto LABEL_8;
    }
    v7 = v5;
    v5 += 40;
    v16 = 1;
    v14[0] = "arg";
    v15 = 3;
    sub_BD6B50(v7, v14);
  }
LABEL_8:
  for ( i = *(_QWORD *)(a3 + 80); a3 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( (*(_BYTE *)(i - 17) & 0x10) == 0 )
    {
      v16 = 1;
      v14[0] = "bb";
      v15 = 3;
      sub_BD6B50((unsigned __int8 *)(i - 24), v14);
    }
    for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      if ( (*(_BYTE *)(j - 17) & 0x10) == 0 && *(_BYTE *)(*(_QWORD *)(j - 16) + 8LL) != 7 )
      {
        v16 = 1;
        v14[0] = "i";
        v15 = 3;
        sub_BD6B50((unsigned __int8 *)(j - 24), v14);
      }
    }
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
