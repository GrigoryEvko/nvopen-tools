// Function: sub_8B4FF0
// Address: 0x8b4ff0
//
_BOOL8 __fastcall sub_8B4FF0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 v7; // r12
  char v9; // al
  char i; // dl
  int v12; // r15d
  int v13; // eax
  int v14; // eax
  __m128i *v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a2;
  v15 = 0;
  v16[0] = 0;
  v9 = *(_BYTE *)(a1 + 140);
  if ( v9 != 12 )
    goto LABEL_5;
  do
  {
    a1 = *(_QWORD *)(a1 + 160);
    v9 = *(_BYTE *)(a1 + 140);
  }
  while ( v9 == 12 );
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
  {
    v7 = *(_QWORD *)(v7 + 160);
LABEL_5:
    ;
  }
  if ( (v9 == 13 || v9 == 6)
    && v9 == i
    && (v9 == 6 ? (v12 = sub_8D46C0(a1), v13 = sub_8D46C0(v7)) : (v12 = sub_8D4870(a1), v13 = sub_8D4870(v7)),
        (a5 & 4) == 0
      ? (a5 |= 8u, v14 = sub_8DF240(v12, v13, 0, 1, 0, 0, (__int64)&v15, (__int64)v16))
      : (v14 = sub_8DF240(v13, v12, 0, 1, 0, 0, (__int64)v16, (__int64)&v15)),
        v14 && (!(unsigned int)sub_8D2310(v15) || dword_4F06978 && (a5 & 4) != 0)) )
  {
    return (unsigned int)sub_8B3500(v15, v16[0], a3, a4, a5) != 0;
  }
  else
  {
    return 0;
  }
}
