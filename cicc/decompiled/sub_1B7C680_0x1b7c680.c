// Function: sub_1B7C680
// Address: 0x1b7c680
//
__int64 __fastcall sub_1B7C680(__int64 a1, __int64 a2)
{
  char v2; // al
  int v3; // ecx
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  int v7; // ecx
  __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // rdx
  _QWORD *v11; // rax
  double v12; // xmm0_8

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 54 )
  {
    v3 = *(unsigned __int16 *)(a2 + 18) >> 1;
    result = (unsigned int)(1 << v3 >> 1);
    if ( 1 << v3 >> 1 )
      return result;
    v5 = *(_QWORD *)(a1 + 40);
    v6 = *(_QWORD *)a2;
    return sub_15A9FE0(v5, v6);
  }
  if ( v2 == 55 )
  {
    v7 = *(unsigned __int16 *)(a2 + 18) >> 1;
    result = (unsigned int)(1 << v7 >> 1);
    if ( !(1 << v7 >> 1) )
      return sub_15A9FE0(*(_QWORD *)(a1 + 40), **(_QWORD **)(a2 - 48));
    return result;
  }
  if ( v2 != 78 )
    return 0;
  v8 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v8 + 16) )
    BUG();
  v9 = *(_DWORD *)(v8 + 36);
  if ( v9 != 4503 && v9 != 4085 )
  {
    if ( v9 != 4057 )
    {
      if ( v9 == 4492 )
      {
        result = sub_15603A0((_QWORD *)(a2 + 56), 2);
        if ( (_DWORD)result )
          return result;
      }
      goto LABEL_15;
    }
    result = sub_15603A0((_QWORD *)(a2 + 56), 1);
    if ( (_DWORD)result )
      return result;
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = pow(2.0, (double)(int)((((unsigned int)v11 >> 13) & 0x1F) - 1));
  result = (unsigned int)(int)v12;
  if ( !(int)v12 )
  {
    if ( v9 != 4085 )
    {
LABEL_15:
      v6 = **(_QWORD **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
LABEL_16:
      v5 = *(_QWORD *)(a1 + 40);
      return sub_15A9FE0(v5, v6);
    }
LABEL_21:
    v6 = *(_QWORD *)a2;
    goto LABEL_16;
  }
  return result;
}
