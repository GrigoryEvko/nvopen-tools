// Function: sub_10C4880
// Address: 0x10c4880
//
__int64 __fastcall sub_10C4880(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  _BYTE *v8; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 54 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
  **(_QWORD **)a1 = v4;
  v6 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v6 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v6 + 24;
    return 1;
  }
  else
  {
    v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v7 <= 1
      && *(_BYTE *)v6 <= 0x15u
      && (v8 = sub_AD7630(v6, *(unsigned __int8 *)(a1 + 16), v7)) != 0
      && *v8 == 17 )
    {
      **(_QWORD **)(a1 + 8) = v8 + 24;
      return 1;
    }
    else
    {
      return 0;
    }
  }
}
