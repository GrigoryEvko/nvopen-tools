// Function: sub_2252B90
// Address: 0x2252b90
//
unsigned __int64 __fastcall sub_2252B90(__int64 a1, __int64 a2)
{
  char v2; // r8
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  unsigned __int64 v6; // [rsp+8h] [rbp-10h] BYREF

  v2 = *(_BYTE *)(a1 + 40);
  if ( v2 == -1 )
  {
    v4 = 0;
    goto LABEL_5;
  }
  v3 = v2 & 7;
  if ( (v2 & 7) == 2 )
  {
    v4 = -2 * a2;
    goto LABEL_5;
  }
  if ( v3 <= 2u )
  {
    if ( v3 )
      goto LABEL_12;
    goto LABEL_7;
  }
  v4 = -4 * a2;
  if ( v3 != 3 )
  {
    if ( v3 != 4 )
LABEL_12:
      abort();
LABEL_7:
    v4 = -8 * a2;
  }
LABEL_5:
  sub_2252A40(*(_BYTE *)(a1 + 40), *(char **)(a1 + 16), (char *)(*(_QWORD *)(a1 + 24) + v4), &v6);
  return v6;
}
