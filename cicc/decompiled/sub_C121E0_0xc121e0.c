// Function: sub_C121E0
// Address: 0xc121e0
//
__int64 __fastcall sub_C121E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v5; // rdx

  v3 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a3 & 4) != 0 )
    return sub_CB6200(a2, *(_QWORD *)v3, *(_QWORD *)(v3 + 8));
  if ( (*(_BYTE *)(v3 + 33) & 3) == 1 )
  {
    v5 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) <= 5 )
    {
      sub_CB6200(a2, "__imp_", 6);
    }
    else
    {
      *(_DWORD *)v5 = 1835622239;
      *(_WORD *)(v5 + 4) = 24432;
      *(_QWORD *)(a2 + 32) += 6LL;
    }
  }
  return sub_E409B0(a1 + 128, a2, v3, 0);
}
