// Function: sub_10E9040
// Address: 0x10e9040
//
__int64 __fastcall sub_10E9040(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rbx
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rax
  unsigned __int8 v6; // dl

  v2 = (_BYTE *)sub_B46B10(a2, 0);
  if ( *v2 == 64 )
  {
    v3 = v2;
    if ( sub_B46220(a2, (__int64)v2) )
      return sub_F207A0(a1, (__int64 *)a2);
    v4 = v3[72];
    if ( *(_BYTE *)(a2 + 72) == v4 && v4 <= 1u && byte_3F8E4E0[8 * (*((_WORD *)v3 + 1) & 7) + (*(_WORD *)(a2 + 2) & 7)] )
      return sub_F207A0(a1, (__int64 *)a2);
  }
  v5 = sub_B46BC0(a2, 0);
  if ( v5
    && *(_BYTE *)v5 == 64
    && (v6 = *(_BYTE *)(v5 + 72), *(_BYTE *)(a2 + 72) == v6)
    && v6 <= 1u
    && byte_3F8E4E0[8 * (*(_WORD *)(v5 + 2) & 7) + (*(_WORD *)(a2 + 2) & 7)] )
  {
    return sub_F207A0(a1, (__int64 *)a2);
  }
  else
  {
    return 0;
  }
}
