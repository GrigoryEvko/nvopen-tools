// Function: sub_16D1C60
// Address: 0x16d1c60
//
__int64 __fastcall sub_16D1C60(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  int v3; // eax
  __int64 *v4; // rax
  __int64 v5; // r8

  v3 = sub_16D1B30((__int64 *)a1, a2, a3);
  if ( v3 == -1 )
    return 0;
  v4 = (__int64 *)(*(_QWORD *)a1 + 8LL * v3);
  v5 = *v4;
  *v4 = -8;
  --*(_DWORD *)(a1 + 12);
  ++*(_DWORD *)(a1 + 16);
  return v5;
}
