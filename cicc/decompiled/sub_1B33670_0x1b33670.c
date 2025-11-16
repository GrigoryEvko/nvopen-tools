// Function: sub_1B33670
// Address: 0x1b33670
//
__int64 __fastcall sub_1B33670(__int64 a1)
{
  unsigned int v1; // r8d
  unsigned __int64 v2; // r12
  int v3; // r13d
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_1AEA030(v5, a1);
  v1 = 1;
  v2 = v5[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v5[0] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return v1;
  if ( (v5[0] & 4) == 0 )
    goto LABEL_12;
  v3 = *(_DWORD *)(v2 + 8);
  if ( *(_QWORD *)v2 != v2 + 16 )
    _libc_free(*(_QWORD *)v2);
  j_j___libc_free_0(v2, 48);
  if ( v3 )
  {
LABEL_12:
    if ( *(_QWORD *)a1 )
    {
      LOBYTE(v1) = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) + 8LL) - 13 > 1;
      return v1;
    }
  }
  return 1;
}
