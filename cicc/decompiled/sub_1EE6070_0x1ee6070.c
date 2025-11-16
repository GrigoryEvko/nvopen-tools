// Function: sub_1EE6070
// Address: 0x1ee6070
//
__int64 __fastcall sub_1EE6070(__int64 a1, _DWORD *a2)
{
  __int64 (*v2)(); // rax
  int v3; // r13d
  unsigned int v4; // r12d
  __int64 result; // rax

  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 )
    BUG();
  v3 = *(_DWORD *)(v2() + 16);
  v4 = v3 + a2[8];
  result = *(unsigned int *)(a1 + 88);
  if ( v4 < *(_DWORD *)(a1 + 88) >> 2 || v4 > (unsigned int)result )
  {
    _libc_free(*(_QWORD *)(a1 + 80));
    result = (__int64)_libc_calloc(v4, 1u);
    if ( !result )
    {
      if ( v4 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        result = 0;
      }
      else
      {
        result = sub_13A3880(1u);
      }
    }
    *(_QWORD *)(a1 + 80) = result;
    *(_DWORD *)(a1 + 88) = v4;
  }
  *(_DWORD *)(a1 + 96) = v3;
  return result;
}
