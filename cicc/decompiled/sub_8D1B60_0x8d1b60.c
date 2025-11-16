// Function: sub_8D1B60
// Address: 0x8d1b60
//
__int64 __fastcall sub_8D1B60(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int8 v5; // al

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u
    && (v2 = *(_QWORD *)(a1 + 168), (*(_BYTE *)(v2 + 112) & 1) != 0) )
  {
    v5 = *(_BYTE *)(v2 + 111);
    *a2 = 1;
    return v5 >> 7;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 40);
    if ( v3 && *(_BYTE *)(v3 + 28) == 3 )
    {
      if ( (unsigned int)sub_736990(a1) )
      {
        *a2 = 1;
        return 1;
      }
      else
      {
        return 0;
      }
    }
    else
    {
      return 0;
    }
  }
}
