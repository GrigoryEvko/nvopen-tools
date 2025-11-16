// Function: sub_27D82A0
// Address: 0x27d82a0
//
__int64 __fastcall sub_27D82A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  char v5; // al
  bool v6; // cc
  unsigned __int64 v7; // rax
  unsigned int v8; // r12d
  unsigned int v10; // eax
  unsigned int v11; // ecx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-28h]
  unsigned __int64 v15; // [rsp+10h] [rbp-20h]
  unsigned int v16; // [rsp+18h] [rbp-18h]

  sub_9AC3E0((__int64)&v13, a2, *(_QWORD *)a1, 0, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), 1);
  v2 = v14;
  if ( v14 > 0x40 )
  {
    v10 = sub_C445E0((__int64)&v13);
    v11 = 32;
    if ( v10 <= 0x20 )
      v11 = v10;
    if ( v2 - 1 <= v11 )
      LOBYTE(v11) = v2 - 1;
    _BitScanReverse64(&v12, 1LL << v11);
    v8 = 63 - (v12 ^ 0x3F);
    if ( v16 <= 0x40 )
      goto LABEL_18;
  }
  else
  {
    _RCX = ~v13;
    if ( v13 == -1 )
    {
      LOBYTE(_RCX) = v14 - 1;
      v5 = 32;
      v6 = v14 - 1 <= 0x20;
    }
    else
    {
      __asm { tzcnt   rcx, rcx }
      if ( (unsigned int)_RCX > 0x20 )
        LODWORD(_RCX) = 32;
      v5 = v14 - 1;
      v6 = (unsigned int)_RCX <= v14 - 1;
    }
    if ( !v6 )
      LOBYTE(_RCX) = v5;
    _BitScanReverse64(&v7, 1LL << _RCX);
    v8 = 63 - (v7 ^ 0x3F);
    if ( v16 <= 0x40 )
      return v8;
  }
  if ( v15 )
  {
    j_j___libc_free_0_0(v15);
    v2 = v14;
  }
  if ( v2 <= 0x40 )
    return v8;
LABEL_18:
  if ( !v13 )
    return v8;
  j_j___libc_free_0_0(v13);
  return v8;
}
