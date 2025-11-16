// Function: sub_3262120
// Address: 0x3262120
//
__int64 __fastcall sub_3262120(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned int v2; // ebx
  unsigned int v3; // r13d
  int v4; // eax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-38h]
  char *v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v1 = a1;
  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 <= 0x40 )
  {
    _RDX = *(char **)a1;
    v3 = 64;
    v10 = *(_DWORD *)(a1 + 8);
    __asm { tzcnt   rax, rdx }
    v9 = (unsigned __int64)_RDX;
    if ( _RDX )
      v3 = _RAX;
    if ( v2 <= v3 )
      v3 = v2;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 8);
    v3 = sub_C44590(a1);
    sub_C43780((__int64)&v9, (const void **)a1);
    v2 = v10;
    if ( v10 > 0x40 )
    {
      sub_C482E0((__int64)&v9, v3);
      v2 = v10;
      if ( v10 > 0x40 )
      {
        v4 = sub_C444A0((__int64)&v9);
        if ( !v4 )
          goto LABEL_5;
        goto LABEL_21;
      }
      goto LABEL_14;
    }
  }
  if ( v3 == v2 )
  {
    v9 = 0;
    v4 = v2;
    goto LABEL_16;
  }
  v9 >>= v3;
LABEL_14:
  if ( v9 )
  {
    _BitScanReverse64(&v8, v9);
    v4 = (v8 ^ 0x3F) + v2 - 64;
  }
  else
  {
    v4 = v2;
  }
LABEL_16:
  if ( !v4 )
  {
    v2 = v10;
    goto LABEL_18;
  }
LABEL_21:
  sub_C44740((__int64)&v11, (char **)&v9, v2 - v4);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  v2 = v12;
  v9 = (unsigned __int64)v11;
  v10 = v12;
LABEL_18:
  v1 = 1;
  if ( !v2 )
    return v1;
  if ( v2 > 0x40 )
  {
LABEL_5:
    LOBYTE(v1) = (unsigned int)sub_C445E0((__int64)&v9) == v2;
    if ( v9 )
      j_j___libc_free_0_0(v9);
    return v1;
  }
  LOBYTE(v1) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v2) == v9;
  return v1;
}
