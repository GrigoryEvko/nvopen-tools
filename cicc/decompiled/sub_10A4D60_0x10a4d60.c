// Function: sub_10A4D60
// Address: 0x10a4d60
//
__int64 __fastcall sub_10A4D60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  char v8; // dl

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 69 )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6 )
  {
    if ( !*(_QWORD *)(v6 + 8) && *(_BYTE *)v5 == 82 )
    {
      v7 = sub_B53900(*(_QWORD *)(a2 - 32));
      sub_B53630(v7, *(_QWORD *)a1);
      if ( !v8 )
        return 0;
      if ( *(_QWORD *)(v5 - 64) != *(_QWORD *)(a1 + 8) )
        return 0;
      v3 = sub_10081F0((__int64 **)(a1 + 16), *(_QWORD *)(v5 - 32));
      if ( !(_BYTE)v3 )
        return 0;
    }
  }
  return v3;
}
