// Function: sub_745C70
// Address: 0x745c70
//
__int64 __fastcall sub_745C70(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rbx
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v8; // rax

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 160);
  if ( !v3 )
    return 0;
  v5 = 0;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v3 + 144) & 4) != 0 )
      goto LABEL_6;
    v6 = *(_QWORD *)(v3 + 120);
    if ( v6 != a2 && !(unsigned int)sub_745900(*(_QWORD *)(v3 + 120), a2) )
    {
      if ( !a3 )
        goto LABEL_6;
      if ( !(unsigned int)sub_8D3410(v6) )
        goto LABEL_6;
      v8 = sub_8D4050(v6);
      if ( !(unsigned int)sub_745900(v8, a2) )
        goto LABEL_6;
    }
    if ( (*(_BYTE *)(v3 + 88) & 3) == 0 )
      return v3;
    if ( v5 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v3 + 88) & 3) < (unsigned __int8)(*(_BYTE *)(v5 + 88) & 3) )
        v5 = v3;
      v3 = *(_QWORD *)(v3 + 112);
      if ( !v3 )
        return v5;
    }
    else
    {
      v5 = v3;
LABEL_6:
      v3 = *(_QWORD *)(v3 + 112);
      if ( !v3 )
        return v5;
    }
  }
}
