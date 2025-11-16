// Function: sub_25DCEB0
// Address: 0x25dceb0
//
_BOOL8 __fastcall sub_25DCEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  unsigned __int64 v10; // r8
  unsigned int v11; // edx
  __int64 v12; // rcx
  unsigned __int8 v13; // al
  _BOOL4 v14; // r15d
  __int64 *v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // [rsp+8h] [rbp-68h]
  _QWORD *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _QWORD v27[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = 20;
  v7 = *(_QWORD *)(a1 + 24);
  v25 = v27;
  v8 = v27;
  v27[0] = v7;
  v26 = 0x400000001LL;
  v9 = 1;
  while ( 1 )
  {
    v10 = v9;
    v11 = v9 - 1;
    v12 = v8[v9 - 1];
    LODWORD(v26) = v9 - 1;
    v13 = *(_BYTE *)(v12 + 8);
    if ( v13 == 16 )
    {
      v22 = v11;
      v23 = *(_QWORD *)(v12 + 24);
      if ( v10 > HIDWORD(v26) )
      {
        sub_C8D5F0((__int64)&v25, v27, v10, 8u, v10, a6);
        v8 = v25;
        v22 = (unsigned int)v26;
      }
      v8[v22] = v23;
      v8 = v25;
      LODWORD(v26) = v26 + 1;
      goto LABEL_18;
    }
    if ( v13 <= 0x10u )
      break;
    if ( (unsigned __int8)(v13 - 17) <= 1u && *(_BYTE *)(*(_QWORD *)(v12 + 24) + 8LL) == 14 )
    {
LABEL_21:
      v14 = 1;
      goto LABEL_22;
    }
LABEL_18:
    if ( !--v6 )
      goto LABEL_21;
    v9 = v26;
    if ( !(_DWORD)v26 )
    {
      v14 = 0;
      goto LABEL_22;
    }
  }
  if ( v13 == 14 )
    goto LABEL_21;
  if ( v13 != 15 )
    goto LABEL_18;
  v14 = (*(_DWORD *)(v12 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(v12 + 8) & 0x100) == 0 )
    goto LABEL_22;
  v15 = *(__int64 **)(v12 + 16);
  v16 = (__int64)&v15[*(unsigned int *)(v12 + 12)];
  if ( v15 == (__int64 *)v16 )
    goto LABEL_18;
  while ( 1 )
  {
    v17 = *v15;
    v18 = *(_BYTE *)(*v15 + 8);
    if ( v18 == 14 )
      break;
    if ( (unsigned __int8)(v18 - 15) > 3u )
    {
      if ( (__int64 *)v16 == ++v15 )
        goto LABEL_15;
    }
    else
    {
      v19 = (unsigned int)v26;
      v20 = (unsigned int)v26 + 1LL;
      if ( v20 > HIDWORD(v26) )
      {
        v24 = v16;
        sub_C8D5F0((__int64)&v25, v27, v20, 8u, v16, a6);
        v19 = (unsigned int)v26;
        v16 = v24;
      }
      ++v15;
      v25[v19] = v17;
      LODWORD(v26) = v26 + 1;
      if ( (__int64 *)v16 == v15 )
      {
LABEL_15:
        v8 = v25;
        goto LABEL_18;
      }
    }
  }
  v8 = v25;
  v14 = 1;
LABEL_22:
  if ( v8 != v27 )
    _libc_free((unsigned __int64)v8);
  return v14;
}
