// Function: sub_E97E50
// Address: 0xe97e50
//
__int64 __fastcall sub_E97E50(__int64 a1)
{
  __int64 v1; // rax
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // r8d
  void (__fastcall *v6)(__int64, __int64); // rax

  v1 = *(unsigned int *)(a1 + 128);
  if ( (unsigned int)v1 <= 1 )
    return 0;
  v2 = *(_DWORD *)(a1 + 128);
  v3 = *(_QWORD *)(a1 + 120) + 32 * v1;
  v4 = *(_QWORD *)(v3 - 64);
  if ( v4 )
  {
    v5 = *(_DWORD *)(v3 - 56);
    if ( *(_DWORD *)(v3 - 24) != v5 || *(_QWORD *)(v3 - 32) != v4 )
    {
      v6 = **(void (__fastcall ***)(__int64, __int64))a1;
      if ( v6 == sub_E97740 )
      {
        *(_QWORD *)(a1 + 288) = v4 + 56;
      }
      else
      {
        ((void (__fastcall *)(__int64, __int64, _QWORD))v6)(a1, v4, v5);
        v2 = *(_DWORD *)(a1 + 128);
      }
    }
  }
  *(_DWORD *)(a1 + 128) = v2 - 1;
  return 1;
}
