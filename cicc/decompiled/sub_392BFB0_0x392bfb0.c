// Function: sub_392BFB0
// Address: 0x392bfb0
//
__int64 __fastcall sub_392BFB0(__int64 a1)
{
  const char *v1; // rsi
  bool v2; // al
  _BYTE *v3; // rdx

  v1 = *(const char **)(a1 + 144);
  *(_QWORD *)(a1 + 104) = v1;
  while ( !sub_392BF20(a1, v1) )
  {
    v2 = sub_392BF70(a1, *(const char **)(a1 + 144));
    v3 = *(_BYTE **)(a1 + 144);
    if ( v2 || *v3 == 13 || *v3 == 10 || v3 == (_BYTE *)(*(_QWORD *)(a1 + 152) + *(_QWORD *)(a1 + 160)) )
      break;
    v1 = v3 + 1;
    *(_QWORD *)(a1 + 144) = v3 + 1;
  }
  return *(_QWORD *)(a1 + 104);
}
