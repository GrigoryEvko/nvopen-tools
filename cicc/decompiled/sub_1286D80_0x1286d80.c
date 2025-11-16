// Function: sub_1286D80
// Address: 0x1286d80
//
__int64 __fastcall sub_1286D80(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v5; // al

  v5 = *(_BYTE *)(a3 + 24);
  if ( v5 == 3 )
  {
    sub_1285D80(a1, a2, a3);
    return a1;
  }
  else if ( v5 > 3u )
  {
    if ( v5 != 20 )
      goto LABEL_9;
    sub_12803A0(a1, (__int64)a2, a3, a4, a5);
    return a1;
  }
  else
  {
    if ( v5 != 1 )
    {
      if ( v5 == 2 )
      {
        sub_1280D90(a1, (__int64)a2, a3);
        return a1;
      }
LABEL_9:
      sub_127B550("cannot generate l-value for this expression!", (_DWORD *)(a3 + 36), 1);
    }
    sub_1286BA0(a1, a2, a3);
    return a1;
  }
}
