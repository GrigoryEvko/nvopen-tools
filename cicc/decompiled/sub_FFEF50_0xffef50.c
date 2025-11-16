// Function: sub_FFEF50
// Address: 0xffef50
//
__int64 __fastcall sub_FFEF50(char *a1, _BYTE *a2, int a3)
{
  char v4; // al
  char v5; // cl
  __int64 v7; // r8
  _BYTE *v8; // rdi
  _BYTE *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rcx

  v4 = *a1;
  v5 = *a2;
  if ( *a1 == 42 && (v7 = *((_QWORD *)a1 - 8)) != 0 && (v8 = (_BYTE *)*((_QWORD *)a1 - 4), *v8 <= 0x15u) && v5 == 44 )
  {
    v9 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v9 > 0x15u || v7 != *((_QWORD *)a2 - 4) )
      return 0;
  }
  else
  {
    if ( v5 != 42 )
      return 0;
    v11 = *((_QWORD *)a2 - 8);
    if ( !v11 )
      return 0;
    v8 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v8 > 0x15u )
      return 0;
    if ( v4 != 44 )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)a1 - 8);
    if ( *v9 > 0x15u || v11 != *((_QWORD *)a1 - 4) )
      return 0;
  }
  if ( v9 != (_BYTE *)sub_AD63D0((__int64)v8) )
    return 0;
  v10 = *((_QWORD *)a1 + 1);
  if ( a3 == 28 )
    return sub_AD6530(v10, (__int64)a2);
  else
    return sub_AD62B0(v10);
}
