// Function: sub_2568AF0
// Address: 0x2568af0
//
__int64 __fastcall sub_2568AF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rax
  unsigned int v5; // r14d
  unsigned __int8 *v6; // rax
  char v7; // al
  unsigned __int8 v8; // dl
  unsigned __int8 v9; // cl
  unsigned __int64 v10; // rax

  v2 = sub_250C680((__int64 *)(a1 + 72));
  if ( v2 )
  {
    v3 = sub_251B1C0(*(_QWORD *)(a2 + 208), *(_QWORD *)(v2 + 24));
    if ( *(_BYTE *)(v3 + 112) || *(_BYTE *)(v3 + 113) )
      return 1;
  }
  v5 = sub_255A010(a1, a2);
  v6 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  v7 = sub_BD5420(v6, *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL));
  v8 = -1;
  v9 = v7;
  v10 = *(_QWORD *)(a1 + 104);
  if ( v10 )
  {
    _BitScanReverse64(&v10, v10);
    v8 = 63 - (v10 ^ 0x3F);
  }
  if ( v8 <= v9 )
    return 1;
  else
    return v5;
}
