// Function: sub_7A6830
// Address: 0x7a6830
//
__int64 __fastcall sub_7A6830(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax
  int v6; // r14d
  __int64 *v7; // rbx
  __int64 v8; // rsi
  unsigned __int64 v9; // rax

  v3 = *(_QWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 96) |= 0x40u;
  v4 = *(_QWORD *)(a2 + 40);
  if ( v3 && (*(_BYTE *)(v3 + 96) & 2) != 0 )
  {
    result = *(_QWORD *)(v4 + 168);
    v6 = 1;
    v7 = *(__int64 **)result;
  }
  else
  {
    result = *(_QWORD *)(v4 + 168);
    v6 = 0;
    v7 = *(__int64 **)(result + 8);
  }
LABEL_4:
  if ( v7 )
  {
    while ( 1 )
    {
      result = *((unsigned __int8 *)v7 + 96);
      if ( (result & 1) == 0 )
        break;
      result = sub_8E5310(v7, *(_QWORD *)(a2 + 56), a2);
      v8 = result;
      if ( !result )
        goto LABEL_17;
      result = *(unsigned __int8 *)(result + 96);
      if ( (result & 2) == 0 )
      {
LABEL_12:
        *(_QWORD *)(v8 + 104) = *(_QWORD *)(a2 + 104) + v7[13];
        result = *(unsigned __int8 *)(v8 + 96) | 0x40u;
        *(_BYTE *)(v8 + 96) |= 0x40u;
        goto LABEL_13;
      }
      if ( *(_QWORD *)(a2 + 24) == v8 )
        goto LABEL_20;
LABEL_13:
      if ( (result & 0x40) != 0 )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 40) + 168LL) + 32LL) + *(_QWORD *)(v8 + 104);
        if ( *(_QWORD *)(a1 + 56) + 1LL < v9 )
          *(_QWORD *)(a1 + 56) = v9 - 1;
        result = sub_7A6830(a1);
      }
LABEL_17:
      if ( !v6 )
      {
LABEL_18:
        v7 = (__int64 *)v7[1];
        goto LABEL_4;
      }
LABEL_8:
      v7 = (__int64 *)*v7;
      if ( !v7 )
        return result;
    }
    if ( !v6 )
      goto LABEL_18;
    if ( (result & 2) == 0 )
      goto LABEL_8;
    result = sub_8E5650(v7);
    v8 = result;
    if ( *(_QWORD *)(a2 + 24) != result || !result )
      goto LABEL_8;
    if ( (*(_BYTE *)(result + 96) & 2) == 0 )
      goto LABEL_12;
LABEL_20:
    *(_QWORD *)(v8 + 104) = *(_QWORD *)(a2 + 104);
    result = *(unsigned __int8 *)(v8 + 96) | 0x40u;
    *(_BYTE *)(v8 + 96) |= 0x40u;
    goto LABEL_13;
  }
  return result;
}
