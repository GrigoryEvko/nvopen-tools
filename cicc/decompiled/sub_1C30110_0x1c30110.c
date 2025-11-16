// Function: sub_1C30110
// Address: 0x1c30110
//
__int64 __fastcall sub_1C30110(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // rdx

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v1 = sub_1625940(a1, "nv.used_bytes_mask", 0x12u);
  result = 0;
  if ( v1 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(v1 - 8LL * *(unsigned int *)(v1 + 8)) + 136LL);
    result = *(_QWORD *)(v3 + 24);
    if ( *(_DWORD *)(v3 + 32) > 0x40u )
      return *(_QWORD *)result;
  }
  return result;
}
