// Function: sub_B53710
// Address: 0xb53710
//
__int64 __fastcall sub_B53710(__int64 a1, __int64 a2)
{
  int v2; // r13d
  int v3; // r12d
  unsigned int v4; // edx
  unsigned int v5; // r8d
  bool v7; // al
  int v8; // eax
  bool v9; // al
  int v10; // eax
  unsigned __int8 v11; // [rsp+8h] [rbp-48h]

  v2 = a2;
  v3 = a1;
  sub_B53630(a1, a2);
  v5 = v4;
  if ( !(_BYTE)v4 )
  {
    if ( BYTE4(a1) && (v7 = sub_B532B0(a2), v5 = 0, v7) )
    {
      v8 = sub_B53550(a1);
      v5 = 0;
      v3 = v8;
    }
    else if ( BYTE4(a2) )
    {
      v11 = v5;
      v9 = sub_B532B0(a1);
      v5 = v11;
      if ( v9 )
      {
        v10 = sub_B53550(a2);
        v5 = v11;
        v2 = v10;
      }
    }
    switch ( v3 )
    {
      case ' ':
        LOBYTE(v5) = ((v2 - 37) & 0xFFFFFFFB) == 0;
        LOBYTE(v4) = (v2 & 0xFFFFFFFB) == 35;
        v5 |= v4;
        break;
      case '"':
        LOBYTE(v5) = (v2 & 0xFFFFFFFD) == 33;
        break;
      case '$':
        LOBYTE(v5) = (v2 & 0xFFFFFFFB) == 33;
        break;
      case '&':
        LOBYTE(v5) = v2 == 39;
        LOBYTE(v4) = v2 == 33;
        v5 |= v4;
        break;
      case '(':
        LOBYTE(v5) = (v2 & 0xFFFFFFF7) == 33;
        break;
      default:
        return v5;
    }
  }
  return v5;
}
