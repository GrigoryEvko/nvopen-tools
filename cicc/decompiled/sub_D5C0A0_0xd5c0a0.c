// Function: sub_D5C0A0
// Address: 0xd5c0a0
//
__int64 __fastcall sub_D5C0A0(_QWORD *a1, unsigned int a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // r13d
  __int64 result; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  int v7; // [rsp+8h] [rbp-28h]

  v2 = *((_DWORD *)a1 + 2);
  if ( a2 >= v2 )
  {
    result = 1;
    if ( a2 == v2 )
      return result;
    goto LABEL_5;
  }
  if ( v2 > 0x40 )
  {
    v3 = v2 - sub_C444A0((__int64)a1);
    goto LABEL_4;
  }
  if ( *a1 )
  {
    _BitScanReverse64(&v5, *a1);
    v3 = 64 - (v5 ^ 0x3F);
LABEL_4:
    result = 0;
    if ( a2 < v3 )
      return result;
  }
LABEL_5:
  sub_C44AB0((__int64)&v6, (__int64)a1, a2);
  if ( *((_DWORD *)a1 + 2) > 0x40u )
  {
    if ( *a1 )
      j_j___libc_free_0_0(*a1);
  }
  *a1 = v6;
  *((_DWORD *)a1 + 2) = v7;
  return 1;
}
