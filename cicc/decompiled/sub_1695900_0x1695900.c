// Function: sub_1695900
// Address: 0x1695900
//
bool __fastcall sub_1695900(__int64 a1)
{
  bool result; // al
  int v2; // r12d
  __int64 *v3; // [rsp-58h] [rbp-58h] BYREF
  __int64 v4; // [rsp-48h] [rbp-48h] BYREF
  int v5; // [rsp-24h] [rbp-24h]

  result = 1;
  if ( !*(_QWORD *)(a1 + 48) )
  {
    sub_16E1010(&v3);
    v2 = v5;
    if ( v3 != &v4 )
      j_j___libc_free_0(v3, v4 + 1);
    result = 0;
    if ( v2 != 3 )
      return (*(_BYTE *)(a1 + 32) & 7) == 1;
  }
  return result;
}
