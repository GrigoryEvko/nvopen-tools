// Function: sub_984B30
// Address: 0x984b30
//
__int64 __fastcall sub_984B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // r9

  v4 = a3;
  v5 = a4;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 18 )
    return sub_9B7FB0(
             *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL),
             *(_QWORD *)(a1 + 72),
             *(_DWORD *)(a1 + 80),
             a2,
             a3,
             a4,
             0);
  if ( *(_DWORD *)(a4 + 8) <= 0x40u && *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    *(_QWORD *)a4 = *(_QWORD *)a2;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a2 + 8);
  }
  else
  {
    sub_C43990(a4, a2);
    v4 = a3;
    v5 = a4;
  }
  if ( *(_DWORD *)(v4 + 8) <= 0x40u && *(_DWORD *)(v5 + 8) <= 0x40u )
  {
    *(_QWORD *)v4 = *(_QWORD *)v5;
    *(_DWORD *)(v4 + 8) = *(_DWORD *)(v5 + 8);
    return 1;
  }
  else
  {
    sub_C43990(v4, v5);
    return 1;
  }
}
