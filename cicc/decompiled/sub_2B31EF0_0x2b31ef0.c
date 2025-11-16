// Function: sub_2B31EF0
// Address: 0x2b31ef0
//
__int64 __fastcall sub_2B31EF0(__int64 a1, __int64 a2, char *a3, __int64 a4, char a5)
{
  __int64 v8; // r9
  int v9; // edi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _BYTE *v18; // rbx
  __int64 v19; // rax
  int v20; // edi
  int v21; // edx
  int v22; // r10d
  _BYTE *v23; // [rsp+8h] [rbp-78h]
  _BYTE *v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h]
  _BYTE v26[96]; // [rsp+20h] [rbp-60h] BYREF

  if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
  {
    v8 = a1 + 96;
    v9 = 3;
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 96);
    v20 = *(_DWORD *)(a1 + 104);
    if ( !v20 )
      return 0;
    v9 = v20 - 1;
  }
  v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v8 + 72LL * v10;
  v12 = *(_QWORD *)v11;
  if ( a2 != *(_QWORD *)v11 )
  {
    v21 = 1;
    while ( v12 != -4096 )
    {
      v22 = v21 + 1;
      v10 = v9 & (v21 + v10);
      v11 = v8 + 72LL * v10;
      v12 = *(_QWORD *)v11;
      if ( a2 == *(_QWORD *)v11 )
        goto LABEL_4;
      v21 = v22;
    }
    return 0;
  }
LABEL_4:
  v24 = v26;
  v25 = 0x600000000LL;
  if ( !*(_DWORD *)(v11 + 16) )
    return 0;
  sub_2B0C870((__int64)&v24, v11 + 8, v11, a4, v12, v8);
  v18 = v24;
  v23 = &v24[8 * (unsigned int)v25];
  if ( v23 == v24 )
  {
LABEL_20:
    if ( v18 != v26 )
      _libc_free((unsigned __int64)v18);
    return 0;
  }
  while ( 1 )
  {
    v13 = *(_QWORD *)v18;
    if ( !a5 )
      goto LABEL_28;
    v19 = *(unsigned int *)(v13 + 120);
    if ( !(_DWORD)v19 )
      v19 = *(unsigned int *)(v13 + 8);
    if ( v19 == a4 )
    {
LABEL_28:
      if ( sub_2B31C30(*(_QWORD *)v18, a3, a4, v15, v16, v17) )
        break;
    }
    v18 += 8;
    if ( v23 == v18 )
    {
      v18 = v24;
      goto LABEL_20;
    }
  }
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
  return v13;
}
