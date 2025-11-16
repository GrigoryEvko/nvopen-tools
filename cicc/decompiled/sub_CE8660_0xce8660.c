// Function: sub_CE8660
// Address: 0xce8660
//
__int64 __fastcall sub_CE8660(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v4; // rdi
  unsigned __int8 v5; // al
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rcx
  _QWORD *v10; // rdx
  _QWORD v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = 0;
  if ( *(_BYTE *)a1 == 22 )
  {
    v1 = sub_B2D680(a1);
    if ( !(_BYTE)v1 )
      return 0;
    if ( !(unsigned __int8)sub_B2BD80(a1) || !sub_B2BE10(a1) )
    {
      v4 = *(_QWORD *)(a1 + 24);
      v11[0] = 0;
      if ( !(unsigned __int8)sub_CE7BB0(v4, "grid_constant", 0xDu, v11) )
        return 0;
      v5 = *(_BYTE *)(v11[0] - 16LL);
      if ( (v5 & 2) != 0 )
      {
        v6 = *(_QWORD *)(v11[0] - 32LL);
        v7 = *(unsigned int *)(v11[0] - 24LL);
      }
      else
      {
        v7 = (*(_WORD *)(v11[0] - 16LL) >> 6) & 0xF;
        v6 = v11[0] - 8LL * ((v5 >> 2) & 0xF) - 16;
      }
      v8 = v6 + 8 * v7;
      if ( v6 == v8 )
        return 0;
      while ( 1 )
      {
        v9 = *(_QWORD *)(*(_QWORD *)v6 + 136LL);
        v10 = *(_QWORD **)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
          v10 = (_QWORD *)*v10;
        if ( (_QWORD *)(unsigned int)(*(_DWORD *)(a1 + 32) + 1) == v10 )
          break;
        v6 += 8;
        if ( v8 == v6 )
          return 0;
      }
    }
  }
  return v1;
}
