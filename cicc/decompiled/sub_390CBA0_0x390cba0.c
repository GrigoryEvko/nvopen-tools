// Function: sub_390CBA0
// Address: 0x390cba0
//
__int64 __fastcall sub_390CBA0(__int64 *a1, __int64 **a2, __int64 a3)
{
  int v3; // ebx
  _QWORD *v4; // r15
  __int64 v5; // r14
  int v6; // eax
  int v8; // eax
  __int64 v9; // [rsp+8h] [rbp-38h]

  v4 = 0;
  v5 = *(_QWORD *)(a3 + 104);
  v9 = a3 + 96;
  if ( v5 == a3 + 96 )
    return 0;
  do
  {
    LOBYTE(v3) = v4 == 0;
    switch ( *(_BYTE *)(v5 + 16) )
    {
      case 4:
        v3 &= sub_390C4C0((__int64)a1, a2, v5);
        goto LABEL_4;
      case 6:
        v3 &= sub_390C800((__int64)a1, a2, v5);
        goto LABEL_4;
      case 7:
        v3 &= sub_390CA60((__int64)a1, a2, v5);
        goto LABEL_4;
      case 8:
        v3 &= sub_390C520((__int64)a1, a2, v5);
        goto LABEL_4;
      case 9:
        v3 &= sub_390C500((__int64)a1, (__int64)a2, v5);
        goto LABEL_4;
      case 0xB:
        LOBYTE(v8) = sub_390CB20(a1, (__int64)a2, v5);
        v3 &= v8;
        goto LABEL_4;
      case 0xC:
        LOBYTE(v6) = sub_390CB60(a1, (__int64)a2, v5);
        v3 &= v6;
LABEL_4:
        if ( (_BYTE)v3 )
          v4 = (_QWORD *)v5;
        break;
      default:
        break;
    }
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v9 != v5 );
  if ( !v4 )
    return 0;
  sub_38CFC60((__int64)a2, v4);
  return 1;
}
