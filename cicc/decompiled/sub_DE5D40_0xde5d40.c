// Function: sub_DE5D40
// Address: 0xde5d40
//
__int64 __fastcall sub_DE5D40(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned int v7; // r12d
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-38h]
  unsigned int *v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  v3 = 1;
  if ( a3 == sub_D970F0((__int64)a1) )
    return v3;
  v5 = sub_DE4F70(a1, a3, a2);
  v6 = sub_DE5CD0(a1, v5);
  sub_DB5510((__int64)&v12, (__int64)a1, (__int64)v6);
  v7 = v13;
  if ( v13 <= 0x40 )
  {
    _RCX = v12;
    if ( v12 )
    {
      _BitScanReverse64(&v11, v12);
      if ( 64 - ((unsigned int)v11 ^ 0x3F) > 0x20 )
      {
        __asm { tzcnt   rcx, rcx }
        if ( (unsigned int)_RCX > v13 )
          LODWORD(_RCX) = v13;
        goto LABEL_5;
      }
    }
LABEL_13:
    sub_C44AB0((__int64)&v14, (__int64)&v12, 0x20u);
    v3 = (unsigned int)v14;
    if ( v15 > 0x40 )
    {
      v3 = *v14;
      j_j___libc_free_0_0(v14);
    }
    v7 = v13;
    goto LABEL_7;
  }
  if ( v7 - (unsigned int)sub_C444A0((__int64)&v12) <= 0x20 )
    goto LABEL_13;
  LODWORD(_RCX) = sub_C44590((__int64)&v12);
LABEL_5:
  v3 = 1 << _RCX;
  if ( (unsigned int)_RCX >= 0x1F )
    v3 = 0x80000000;
LABEL_7:
  if ( v7 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return v3;
}
