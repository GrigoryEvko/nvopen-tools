// Function: sub_1E2EA70
// Address: 0x1e2ea70
//
__int64 *__fastcall sub_1E2EA70(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = *(_QWORD *)(a1 + 1736);
  if ( !v3 )
  {
    v3 = sub_22077B0(96);
    if ( v3 )
    {
      *(_QWORD *)(v3 + 8) = 0;
      *(_QWORD *)v3 = a1 + 168;
      *(_QWORD *)(v3 + 16) = 0;
      *(_QWORD *)(v3 + 24) = 0;
      *(_DWORD *)(v3 + 32) = 0;
      *(_QWORD *)(v3 + 40) = 0;
      *(_QWORD *)(v3 + 48) = 0;
      *(_QWORD *)(v3 + 56) = 0;
      *(_QWORD *)(v3 + 64) = 0;
      *(_QWORD *)(v3 + 72) = 0;
      *(_QWORD *)(v3 + 80) = 0;
      *(_DWORD *)(v3 + 88) = 0;
    }
    *(_QWORD *)(a1 + 1736) = v3;
  }
  return sub_1E2E5F0(v3, a2);
}
