// Function: sub_10C3DB0
// Address: 0x10c3db0
//
__int64 __fastcall sub_10C3DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  _BYTE *v9; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 69 )
    return 0;
  v4 = *(_BYTE **)(a2 - 32);
  if ( *v4 != 56 )
    return 0;
  v5 = *((_QWORD *)v4 - 8);
  if ( !v5 )
    return 0;
  **(_QWORD **)a1 = v5;
  v7 = *((_QWORD *)v4 - 4);
  if ( *(_BYTE *)v7 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v7 + 24;
    return 1;
  }
  else
  {
    v8 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v8 <= 1
      && *(_BYTE *)v7 <= 0x15u
      && (v9 = sub_AD7630(v7, *(unsigned __int8 *)(a1 + 16), v8)) != 0
      && *v9 == 17 )
    {
      **(_QWORD **)(a1 + 8) = v9 + 24;
      return 1;
    }
    else
    {
      return 0;
    }
  }
}
