// Function: sub_101A620
// Address: 0x101a620
//
__int64 __fastcall sub_101A620(
        unsigned int a1,
        __int64 ***a2,
        unsigned __int8 *a3,
        char a4,
        __m128i *a5,
        unsigned int a6)
{
  __int64 v9; // r12
  _BYTE *v11; // r13
  unsigned int v12; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  unsigned int v20; // r15d
  unsigned int v21; // r14d
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // eax
  unsigned int v27; // edx
  unsigned __int8 *v30; // [rsp+0h] [rbp-60h] BYREF
  __int64 ***v31; // [rsp+8h] [rbp-58h] BYREF
  __int64 v32; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+28h] [rbp-38h]

  v31 = a2;
  v30 = a3;
  v9 = sub_FFE3E0(a1, (_BYTE **)&v31, &v30, a5->m128i_i64);
  if ( v9 )
    return v9;
  v9 = (__int64)sub_1019820(a1, (__int64)v31, v30, a5, a6);
  if ( v9 || !a4 )
    return v9;
  v11 = v30 + 24;
  if ( *v30 != 17 )
  {
    v16 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v30 + 1) + 8LL) - 17;
    if ( (unsigned int)v16 > 1 )
      return v9;
    if ( *v30 > 0x15u )
      return v9;
    v17 = sub_AD7630((__int64)v30, 0, v16);
    if ( !v17 || *v17 != 17 )
      return v9;
    v11 = v17 + 24;
  }
  v12 = *((_DWORD *)v11 + 2);
  if ( v12 <= 0x40 )
  {
    _RDX = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 )
    {
      __asm { tzcnt   rcx, rdx }
      if ( (unsigned int)_RCX <= v12 )
        v12 = _RCX;
      if ( !v12 )
      {
LABEL_29:
        if ( (_RDX & (_RDX - 1)) == 0 )
          return v9;
        goto LABEL_9;
      }
    }
    else if ( !v12 )
    {
      goto LABEL_9;
    }
  }
  else if ( !(unsigned int)sub_C44590((__int64)v11) )
  {
LABEL_8:
    if ( (unsigned int)sub_C44630((__int64)v11) == 1 )
      return v9;
    goto LABEL_9;
  }
  sub_9AC330((__int64)&v32, (__int64)v31, 0, a5);
  v20 = v35;
  if ( v35 <= 0x40 )
  {
    _RAX = v34;
    v21 = 64;
    __asm { tzcnt   rdx, rax }
    if ( v34 )
      v21 = _RDX;
    if ( v35 <= v21 )
      v21 = v35;
  }
  else
  {
    v21 = sub_C44590((__int64)&v34);
  }
  v22 = *((_DWORD *)v11 + 2);
  if ( v22 <= 0x40 )
  {
    _RDX = *(_QWORD *)v11;
    __asm { tzcnt   rcx, rdx }
    v27 = 64;
    if ( *(_QWORD *)v11 )
      v27 = _RCX;
    if ( v22 > v27 )
      v22 = v27;
  }
  else
  {
    v22 = sub_C44590((__int64)v11);
  }
  if ( v22 > v21 )
  {
    v9 = sub_ACADE0(v31[1]);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    return v9;
  }
  if ( v20 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( *((_DWORD *)v11 + 2) > 0x40u )
    goto LABEL_8;
  _RDX = *(_QWORD *)v11;
  if ( *(_QWORD *)v11 )
    goto LABEL_29;
LABEL_9:
  v13 = *(unsigned __int8 *)v31;
  if ( a1 == 19 )
  {
    if ( (unsigned __int8)v13 <= 0x1Cu )
    {
      if ( (_BYTE)v13 != 5 )
        return v9;
      v24 = *((unsigned __int16 *)v31 + 1);
      if ( (*((_WORD *)v31 + 1) & 0xFFF7) != 0x11 && (v24 & 0xFFFD) != 0xD )
        return v9;
    }
    else
    {
      if ( (unsigned __int8)v13 > 0x36u )
        return v9;
      v23 = 0x40540000000000LL;
      if ( !_bittest64(&v23, v13) )
        return v9;
      v24 = (unsigned __int8)v13 - 29;
    }
    if ( v24 != 17 || (*((_BYTE *)v31 + 1) & 4) == 0 )
      return v9;
  }
  else
  {
    if ( (unsigned __int8)v13 <= 0x1Cu )
    {
      if ( (_BYTE)v13 != 5 )
        return v9;
      v15 = *((unsigned __int16 *)v31 + 1);
      if ( (*((_WORD *)v31 + 1) & 0xFFF7) != 0x11 && (v15 & 0xFFFD) != 0xD )
        return v9;
    }
    else
    {
      if ( (unsigned __int8)v13 > 0x36u )
        return v9;
      v14 = 0x40540000000000LL;
      if ( !_bittest64(&v14, v13) )
        return v9;
      v15 = (unsigned __int8)v13 - 29;
    }
    if ( v15 != 17 || (*((_BYTE *)v31 + 1) & 2) == 0 )
      return v9;
  }
  if ( *(v31 - 8) && v30 == (unsigned __int8 *)*(v31 - 4) )
    return (__int64)*(v31 - 8);
  return v9;
}
