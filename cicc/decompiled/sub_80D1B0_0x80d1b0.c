// Function: sub_80D1B0
// Address: 0x80d1b0
//
__int64 __fastcall sub_80D1B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx
  unsigned int v4; // [rsp-14h] [rbp-14h]

  if ( (*(_BYTE *)(a1 + 176) & 0x20) != 0 )
  {
    result = 1;
  }
  else if ( *(_QWORD *)(a1 + 8) )
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x30 )
      return result;
    result = 1;
    if ( (*(_BYTE *)(a1 + 170) & 0x10) == 0 && (*(_BYTE *)(a1 + 89) & 4) == 0 )
    {
      v2 = *(_QWORD *)(a1 + 40);
      result = 0;
      if ( v2 )
        result = *(_BYTE *)(v2 + 28) == 3;
    }
  }
  else
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 170) & 2) == 0 )
      return result;
    result = 1;
    if ( *(_BYTE *)(a1 + 136) == 3 )
      return 0;
  }
  if ( unk_4D04170 )
  {
    v3 = *(_QWORD *)(a1 + 104);
    if ( !v3 || (*(_BYTE *)(v3 + 11) & 0x20) == 0 )
    {
      v4 = result;
      sub_80A450(a1, 7u);
      result = v4;
    }
    if ( *(char *)(a1 + 168) < 0 )
      return 1;
  }
  return result;
}
