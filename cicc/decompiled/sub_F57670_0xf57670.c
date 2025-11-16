// Function: sub_F57670
// Address: 0xf57670
//
__int64 __fastcall sub_F57670(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r14
  unsigned int v4; // r12d
  __int64 v6; // r15
  __int64 v7; // rsi
  int v8; // eax
  unsigned int v9[13]; // [rsp+Ch] [rbp-34h] BYREF

  v2 = (_QWORD *)(a1 + 72);
  v4 = sub_A747A0((_QWORD *)(a1 + 72), "gc-leaf-function", 0x10u);
  if ( (_BYTE)v4 )
    return v4;
  v4 = sub_B49590(a1, "gc-leaf-function", 0x10u);
  if ( (_BYTE)v4 )
    return 1;
  v6 = *(_QWORD *)(a1 - 32);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a1 + 80) )
    goto LABEL_7;
  if ( (unsigned __int8)sub_B2D620(*(_QWORD *)(a1 - 32), "gc-leaf-function", 0x10u) )
    return 1;
  v8 = *(_DWORD *)(v6 + 36);
  if ( v8 )
  {
    if ( v8 == 151 || v8 == 146 || v8 == 239 || v8 == 242 )
      return v4;
    return 1;
  }
LABEL_7:
  if ( !(unsigned __int8)sub_A73ED0(v2, 23) && !(unsigned __int8)sub_B49560(a1, 23)
    || (unsigned __int8)sub_A73ED0(v2, 4)
    || (unsigned __int8)sub_B49560(a1, 4) )
  {
    v7 = *(_QWORD *)(a1 - 32);
    if ( v7
      && !*(_BYTE *)v7
      && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a1 + 80)
      && sub_981210(*a2, v7, v9)
      && (a2[((unsigned __int64)v9[0] >> 6) + 1] & (1LL << SLOBYTE(v9[0]))) == 0 )
    {
      LOBYTE(v4) = (((int)*(unsigned __int8 *)(*a2 + (v9[0] >> 2)) >> (2 * (v9[0] & 3))) & 3) != 0;
    }
  }
  return v4;
}
