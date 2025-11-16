// Function: sub_14B2B20
// Address: 0x14b2b20
//
char __fastcall sub_14B2B20(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    if ( !v4 || (**a1 = v4, !(unsigned __int8)sub_14A9710(*(_QWORD *)(a2 - 24))) )
    {
      v5 = *(_QWORD *)(a2 - 24);
      if ( v5 )
      {
        **a1 = v5;
        return sub_14A9710(*(_QWORD *)(a2 - 48));
      }
      return 0;
    }
    return 1;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 - 24 * v6);
  if ( v7 )
  {
    **a1 = v7;
    if ( !sub_14A9880(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    {
      v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      goto LABEL_13;
    }
    return 1;
  }
LABEL_13:
  v8 = *(_QWORD *)(a2 + 24 * (1 - v6));
  if ( !v8 )
    return 0;
  **a1 = v8;
  return sub_14A9880(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
}
