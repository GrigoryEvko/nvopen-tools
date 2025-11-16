// Function: sub_15E3650
// Address: 0x15e3650
//
__int64 __fastcall sub_15E3650(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  char v4; // dl
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx

  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return sub_1C2FA50(a1);
  while ( 1 )
  {
    v3 = sub_1648700(v2);
    v4 = *(_BYTE *)(v3 + 16);
    if ( v4 != 4 )
    {
      if ( v4 == 78 )
      {
        v6 = v3 | 4;
      }
      else
      {
        if ( v4 != 29 )
          break;
        v6 = v3 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      v8 = (v6 & 4) != 0 ? v7 - 24 : v7 - 72;
      if ( v2 != v8 )
        break;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return sub_1C2FA50(a1);
  }
  if ( a2 )
    *a2 = v3;
  return 1;
}
