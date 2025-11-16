// Function: sub_3444E20
// Address: 0x3444e20
//
__int64 __fastcall sub_3444E20(__int64 **a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // r13d
  const void **v13; // r12
  __int64 *v14; // r14
  __int64 v16; // rax
  int v17; // ebx
  unsigned int v18; // ebx
  unsigned __int64 v19; // r12
  int v20; // edx
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  int v24; // edx
  unsigned __int64 v25; // rax
  unsigned __int64 v27; // rdx
  unsigned int v28; // [rsp-6Ch] [rbp-6Ch]
  unsigned int v29; // [rsp-6Ch] [rbp-6Ch]
  unsigned __int64 v30; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v31; // [rsp-60h] [rbp-60h]
  _QWORD *v32; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v33; // [rsp-50h] [rbp-50h]
  _QWORD *v34; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v35; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 24) != 58 )
    return 0;
  v7 = *(_QWORD *)(a2 + 56);
  if ( !v7 )
    return 0;
  v8 = 1;
  do
  {
    while ( a3 != *(_DWORD *)(v7 + 8) )
    {
      v7 = *(_QWORD *)(v7 + 32);
      if ( !v7 )
        goto LABEL_11;
    }
    if ( !v8 )
      return 0;
    v9 = *(_QWORD *)(v7 + 32);
    if ( !v9 )
      goto LABEL_12;
    if ( a3 == *(_DWORD *)(v9 + 8) )
      return 0;
    v7 = *(_QWORD *)(v9 + 32);
    v8 = 0;
  }
  while ( v7 );
LABEL_11:
  if ( v8 == 1 )
    return 0;
LABEL_12:
  v10 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), 0, 0, a5, a6);
  if ( !v10 || (*(_BYTE *)(v10 + 32) & 8) != 0 )
    return 0;
  v11 = *(_QWORD *)(v10 + 96);
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 <= 0x40 )
  {
    v22 = *(_QWORD *)(v11 + 24);
    if ( !v22 || (v22 & (v22 - 1)) == 0 )
      return 0;
    v35 = v12;
    v14 = *a1;
    v34 = (_QWORD *)v22;
    goto LABEL_35;
  }
  v13 = (const void **)(v11 + 24);
  if ( v12 == (unsigned int)sub_C444A0(v11 + 24) || (unsigned int)sub_C44630((__int64)v13) == 1 )
    return 0;
  v35 = v12;
  v14 = *a1;
  sub_C43780((__int64)&v34, v13);
  v12 = v35;
  if ( v35 <= 0x40 )
  {
    v22 = (unsigned __int64)v34;
LABEL_35:
    v23 = *v14 | v22;
    v31 = v12;
    v30 = v23;
    _R12 = v23;
    v16 = 1LL << ((unsigned __int8)v12 - 1);
    goto LABEL_36;
  }
  sub_C43BD0(&v34, v14);
  v12 = v35;
  _R12 = (unsigned __int64)v34;
  v31 = v35;
  v30 = (unsigned __int64)v34;
  v16 = 1LL << ((unsigned __int8)v35 - 1);
  if ( v35 <= 0x40 )
  {
LABEL_36:
    if ( (_R12 & v16) != 0 )
    {
      if ( v12 )
      {
        v24 = 64;
        if ( _R12 << (64 - (unsigned __int8)v12) != -1 )
        {
          _BitScanReverse64(&v25, ~(_R12 << (64 - (unsigned __int8)v12)));
          v24 = v25 ^ 0x3F;
        }
      }
      else
      {
        v24 = 0;
      }
      __asm { tzcnt   rax, r12 }
      if ( (unsigned int)_RAX > v12 )
        LODWORD(_RAX) = v12;
      if ( v24 + (_DWORD)_RAX == v12 )
      {
        v33 = v12;
LABEL_44:
        v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
        if ( !v12 )
          v27 = 0;
        v32 = (_QWORD *)(v27 & ~_R12);
        goto LABEL_25;
      }
    }
    return 0;
  }
  if ( (v34[(v35 - 1) >> 6] & v16) == 0
    || (v17 = sub_C44500((__int64)&v30), v12 != v17 + (unsigned int)sub_C44590((__int64)&v30)) )
  {
    if ( _R12 )
      j_j___libc_free_0_0(_R12);
    return 0;
  }
  v33 = v12;
  sub_C43780((__int64)&v32, (const void **)&v30);
  v12 = v33;
  if ( v33 <= 0x40 )
  {
    _R12 = (unsigned __int64)v32;
    goto LABEL_44;
  }
  sub_C43D10((__int64)&v32);
LABEL_25:
  sub_C46250((__int64)&v32);
  v18 = v33;
  v19 = (unsigned __int64)v32;
  v33 = 0;
  v35 = v18;
  v34 = v32;
  if ( v18 > 0x40 )
  {
    result = v18 - 1 - (unsigned int)sub_C444A0((__int64)&v34);
    if ( v19 )
    {
      v29 = result;
      j_j___libc_free_0_0(v19);
      result = v29;
      if ( v33 > 0x40 )
      {
        if ( v32 )
        {
          j_j___libc_free_0_0((unsigned __int64)v32);
          result = v29;
        }
      }
    }
  }
  else
  {
    v20 = 64;
    if ( v32 )
    {
      _BitScanReverse64(&v21, (unsigned __int64)v32);
      v20 = v21 ^ 0x3F;
    }
    result = (unsigned int)(63 - v20);
  }
  if ( v31 > 0x40 )
  {
    if ( v30 )
    {
      v28 = result;
      j_j___libc_free_0_0(v30);
      return v28;
    }
  }
  return result;
}
