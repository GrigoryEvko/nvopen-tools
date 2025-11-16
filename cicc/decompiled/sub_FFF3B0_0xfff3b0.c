// Function: sub_FFF3B0
// Address: 0xfff3b0
//
__int64 __fastcall sub_FFF3B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rax

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **(_QWORD **)a1 = v2;
  v4 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v4 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v4 + 24;
    return 1;
  }
  else
  {
    v6 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
    if ( (unsigned int)v6 <= 1
      && *(_BYTE *)v4 <= 0x15u
      && (v7 = sub_AD7630(v4, *(unsigned __int8 *)(a1 + 16), v6)) != 0
      && *v7 == 17 )
    {
      **(_QWORD **)(a1 + 8) = v7 + 24;
      return 1;
    }
    else
    {
      return 0;
    }
  }
}
