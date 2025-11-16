// Function: sub_140B890
// Address: 0x140b890
//
__int64 __fastcall sub_140B890(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  unsigned int v3; // r13d
  unsigned int v4; // r12d
  __int64 result; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  int v8; // [rsp+8h] [rbp-28h]

  v2 = *((_DWORD *)a2 + 2);
  v3 = *(_DWORD *)(a1 + 20);
  if ( v3 >= v2 )
  {
    result = 1;
    if ( v3 == v2 )
      return result;
    goto LABEL_5;
  }
  if ( v2 > 0x40 )
  {
    v4 = v2 - sub_16A57B0(a2);
    goto LABEL_4;
  }
  if ( *a2 )
  {
    _BitScanReverse64(&v6, *a2);
    v4 = 64 - (v6 ^ 0x3F);
LABEL_4:
    result = 0;
    if ( v3 < v4 )
      return result;
  }
LABEL_5:
  sub_16A5D10(&v7, a2, v3);
  if ( *((_DWORD *)a2 + 2) > 0x40u )
  {
    if ( *a2 )
      j_j___libc_free_0_0(*a2);
  }
  *a2 = v7;
  *((_DWORD *)a2 + 2) = v8;
  return 1;
}
