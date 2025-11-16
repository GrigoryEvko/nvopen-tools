// Function: sub_72F790
// Address: 0x72f790
//
__int64 __fastcall sub_72F790(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 i; // rax
  _DWORD v6[5]; // [rsp+Ch] [rbp-14h] BYREF

  if ( *(_BYTE *)(a1 + 174) == 5
    && *(_BYTE *)(a1 + 176) == 15
    && (unsigned int)sub_72F5E0(*(_QWORD *)(a1 + 152), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), 1, 0, a2, v6)
    && !v6[0] )
  {
    for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    *a3 = sub_8D3110(*(_QWORD *)(**(_QWORD **)(i + 168) + 8LL));
    return 1;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
