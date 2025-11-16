// Function: sub_B46C60
// Address: 0xb46c60
//
__int64 __fastcall sub_B46C60(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax

  if ( *(_BYTE *)a1 == 85
    && (v2 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v2
    && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80)
    && (*(_BYTE *)(v2 + 33) & 0x20) != 0
    && (unsigned int)(*(_DWORD *)(v2 + 36) - 68) <= 3
    && (v3 = sub_B46B10(a1, 0)) != 0 )
  {
    return v3 + 48;
  }
  else
  {
    return a1 + 48;
  }
}
