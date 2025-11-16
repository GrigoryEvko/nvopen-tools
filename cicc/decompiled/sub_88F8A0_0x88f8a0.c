// Function: sub_88F8A0
// Address: 0x88f8a0
//
__int64 __fastcall sub_88F8A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // rcx
  char v5; // dl
  __int64 v6; // rbx
  char v8; // al

  v2 = a1;
  if ( a1 )
  {
    if ( (*(_WORD *)(a1 + 80) & 0x10FF) == 0x1010 )
    {
      sub_6851C0(0x12Au, (_DWORD *)(a2 + 8));
      v8 = *(_BYTE *)(a1 + 80);
      if ( v8 == 16 )
      {
        v2 = **(_QWORD **)(a1 + 88);
        v8 = *(_BYTE *)(v2 + 80);
      }
      if ( v8 == 24 )
        return *(_QWORD *)(v2 + 88);
    }
    else
    {
      v3 = *(_BYTE *)(a1 + 80);
      if ( (v3 == 24 || (*(_BYTE *)(a1 + 82) & 8) != 0) && (*(_BYTE *)(a2 + 16) & 4) == 0 )
      {
        v4 = 0;
        if ( (*(_BYTE *)(a2 + 18) & 2) == 0 )
          v4 = *(_QWORD *)(a2 + 32);
        v5 = *(_BYTE *)(a1 + 80);
        v6 = a1;
        if ( v3 == 16 )
        {
          v6 = **(_QWORD **)(a1 + 88);
          v5 = *(_BYTE *)(v6 + 80);
        }
        if ( v5 == 24 )
          v6 = *(_QWORD *)(v6 + 88);
        if ( (*(_BYTE *)(a1 + 82) & 8) == 0
          || (v2 = v6, v3 != 17) && !(unsigned int)sub_880800(v6, *(_QWORD *)(v4 + 128)) )
        {
          v2 = v6;
          sub_887650(a2);
        }
      }
    }
  }
  return v2;
}
