// Function: sub_129F080
// Address: 0x129f080
//
void __fastcall sub_129F080(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // edi
  __int16 v4; // ax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rsi
  unsigned int v9; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_DWORD *)(a1 + 480);
  if ( v3 )
  {
    v4 = *(_WORD *)(a1 + 484);
    if ( v4 )
    {
      if ( !*a2 || v3 != *(_DWORD *)(a1 + 488) || v4 != *(_WORD *)(a1 + 492) )
      {
        *(_DWORD *)(a1 + 488) = *(_DWORD *)(a1 + 480);
        *(_WORD *)(a1 + 492) = *(_WORD *)(a1 + 484);
        v6 = *(_QWORD *)(a1 + 544);
        if ( v6 == *(_QWORD *)(a1 + 552) )
          v6 = *(_QWORD *)(*(_QWORD *)(a1 + 568) - 8LL) + 512LL;
        v7 = *(_QWORD *)(v6 - 8);
        sub_129E300(v3, (char *)&v9);
        sub_15C7110(v10, v9, *(unsigned __int16 *)(a1 + 484), v7, 0);
        if ( *a2 )
          sub_161E7C0(a2);
        v8 = v10[0];
        *a2 = v10[0];
        if ( v8 )
          sub_1623210(v10, v8, a2);
      }
    }
  }
}
