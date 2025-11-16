// Function: sub_C7AF00
// Address: 0xc7af00
//
__int64 __fastcall sub_C7AF00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  unsigned int v4; // r14d
  unsigned __int64 v8; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // edx
  unsigned __int64 v12; // rax
  const void *v13; // rax
  unsigned int v14; // eax
  unsigned __int64 v15; // rcx
  const void *v16; // rcx
  bool v17; // cc
  int v18; // [rsp+Ch] [rbp-84h]
  __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-78h]
  const void *v21; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-68h]
  const void *v23; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-58h]
  __int64 v25; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-48h]
  const void *v27; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+58h] [rbp-38h]

  v3 = *(_DWORD *)(a3 + 8);
  v4 = *(_DWORD *)(a2 + 8);
  if ( !v3 )
    goto LABEL_5;
  if ( v3 <= 0x40 )
  {
    v8 = *(_QWORD *)a3;
    if ( *(_QWORD *)a3 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) || (v8 & 1) == 0 )
      goto LABEL_5;
    _RDX = ~v8;
    __asm { tzcnt   rax, rax }
    if ( !_RDX )
    {
      v20 = *(_DWORD *)(a2 + 8);
      if ( v4 <= 0x40 )
      {
        v19 = 0;
        v10 = -1;
LABEL_46:
        v19 |= v10;
        goto LABEL_18;
      }
      sub_C43690((__int64)&v19, 0, 0);
      v10 = -1;
LABEL_16:
      if ( v20 > 0x40 )
      {
        *(_QWORD *)v19 |= v10;
        goto LABEL_18;
      }
      goto LABEL_46;
    }
  }
  else
  {
    LODWORD(_RAX) = sub_C445E0(a3);
    if ( v3 == (_DWORD)_RAX || (**(_BYTE **)a3 & 1) == 0 )
    {
LABEL_5:
      *(_DWORD *)(a1 + 8) = v4;
      if ( v4 > 0x40 )
      {
        sub_C43690(a1, 0, 0);
        *(_DWORD *)(a1 + 24) = v4;
        sub_C43690(a1 + 16, 0, 0);
      }
      else
      {
        *(_QWORD *)a1 = 0;
        *(_DWORD *)(a1 + 24) = v4;
        *(_QWORD *)(a1 + 16) = 0;
      }
      return a1;
    }
  }
  v20 = v4;
  if ( v4 > 0x40 )
  {
    v18 = _RAX;
    sub_C43690((__int64)&v19, 0, 0);
    LODWORD(_RAX) = v18;
  }
  else
  {
    v19 = 0;
  }
  if ( (_DWORD)_RAX )
  {
    if ( (unsigned int)_RAX <= 0x40 )
    {
      v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)_RAX);
      goto LABEL_16;
    }
    sub_C43C90(&v19, 0, _RAX);
  }
LABEL_18:
  v11 = *(_DWORD *)(a2 + 24);
  v28 = v11;
  if ( v11 <= 0x40 )
  {
    v12 = *(_QWORD *)(a2 + 16);
LABEL_20:
    v13 = (const void *)(v19 & v12);
    goto LABEL_21;
  }
  sub_C43780((__int64)&v27, (const void **)(a2 + 16));
  v11 = v28;
  if ( v28 <= 0x40 )
  {
    v12 = (unsigned __int64)v27;
    goto LABEL_20;
  }
  sub_C43B90(&v27, &v19);
  v11 = v28;
  v13 = v27;
LABEL_21:
  v21 = v13;
  v14 = *(_DWORD *)(a2 + 8);
  v22 = v11;
  v28 = v14;
  if ( v14 > 0x40 )
  {
    sub_C43780((__int64)&v27, (const void **)a2);
    v14 = v28;
    if ( v28 > 0x40 )
    {
      sub_C43B90(&v27, &v19);
      v14 = v28;
      v16 = v27;
      v11 = v22;
      goto LABEL_24;
    }
    v15 = (unsigned __int64)v27;
    v11 = v22;
  }
  else
  {
    v15 = *(_QWORD *)a2;
  }
  v16 = (const void *)(v19 & v15);
LABEL_24:
  v24 = v14;
  v23 = v16;
  v28 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v27, &v21);
    v14 = v24;
  }
  else
  {
    v27 = v21;
  }
  v26 = v14;
  if ( v14 > 0x40 )
  {
    sub_C43780((__int64)&v25, &v23);
    v17 = v24 <= 0x40;
    *(_DWORD *)(a1 + 8) = v26;
    *(_QWORD *)a1 = v25;
    *(_DWORD *)(a1 + 24) = v28;
    *(_QWORD *)(a1 + 16) = v27;
    if ( !v17 && v23 )
      j_j___libc_free_0_0(v23);
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v14;
    *(_QWORD *)a1 = v23;
    *(_DWORD *)(a1 + 24) = v28;
    *(_QWORD *)(a1 + 16) = v27;
  }
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
