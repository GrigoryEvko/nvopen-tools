// Function: sub_2E31210
// Address: 0x2e31210
//
__int64 __fastcall sub_2E31210(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 (*v4)(void); // rax
  __int64 (*v5)(); // rax

  v2 = 0;
  v3 = a2;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 128LL);
  if ( v4 != sub_2DAC790 )
    v2 = v4();
  if ( a1 + 48 != a2 )
  {
    do
    {
      if ( *(_WORD *)(v3 + 68) != 68 && *(_WORD *)(v3 + 68) != 0 && (unsigned __int16)(*(_WORD *)(v3 + 68) - 3) > 3u )
      {
        v5 = *(__int64 (**)())(*(_QWORD *)v2 + 1336LL);
        if ( v5 == sub_2E2F9B0 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v5)(v2, v3, 0) )
          break;
      }
      if ( (*(_BYTE *)v3 & 4) == 0 && (*(_BYTE *)(v3 + 44) & 8) != 0 )
      {
        do
          v3 = *(_QWORD *)(v3 + 8);
        while ( (*(_BYTE *)(v3 + 44) & 8) != 0 );
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( a1 + 48 != v3 );
  }
  return v3;
}
