// Function: sub_E5B630
// Address: 0xe5b630
//
__int64 __fastcall sub_E5B630(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned __int8 v4; // [rsp+Fh] [rbp-11h]

  result = sub_E97E50();
  if ( (_BYTE)result )
  {
    v2 = *(unsigned int *)(a1 + 128);
    v4 = result;
    if ( !(_DWORD)v2 )
      BUG();
    v3 = *(_QWORD *)(a1 + 120) + 32 * v2 - 32;
    (***(void (__fastcall ****)(_QWORD, _QWORD, __int64, _QWORD, _QWORD))v3)(
      *(_QWORD *)v3,
      *(_QWORD *)(a1 + 312),
      *(_QWORD *)(a1 + 8) + 24LL,
      *(_QWORD *)(a1 + 304),
      *(unsigned int *)(v3 + 8));
    return v4;
  }
  return result;
}
