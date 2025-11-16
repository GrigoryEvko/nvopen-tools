// Function: sub_E5EFF0
// Address: 0xe5eff0
//
char __fastcall sub_E5EFF0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r14
  char v6; // cl
  __int64 v7; // r12
  unsigned __int64 v8; // rax

  switch ( *(_BYTE *)(a2 + 28) )
  {
    case 4:
      return sub_E5E4A0((__int64)a1, a2);
    case 6:
      return sub_E5EB60(a1, a2);
    case 7:
      return sub_E5EC20(a1, a2);
    case 8:
      return sub_E5E620(a1, a2);
    case 9:
      if ( *(_QWORD *)(a2 + 32) )
      {
        v2 = 0;
        v3 = sub_E5C2C0((__int64)a1, a2);
        v4 = *(_QWORD **)a2;
        v5 = v3;
        while ( 1 )
        {
          v2 += sub_E5BD20(a1, (__int64)v4);
          if ( v4 == *(_QWORD **)(a2 + 32) )
            break;
          v4 = (_QWORD *)*v4;
        }
        v6 = *(_BYTE *)(a2 + 30);
        v7 = v5 + v2;
        if ( v5 >> v6 == (unsigned __int64)(v7 - 1) >> v6 && (v8 = 0, (v7 & ~(-1LL << v6)) != 0) )
        {
          if ( !*(_QWORD *)(a2 + 40) )
            return 0;
        }
        else
        {
          v8 = (-(1LL << v6) & (v5 + (1LL << v6) - 1)) - v5;
          if ( *(_QWORD *)(a2 + 40) == v8 )
            return 0;
        }
        *(_QWORD *)(a2 + 40) = v8;
        return 1;
      }
      return 0;
    case 0xB:
      return sub_E5ED30(a1, a2);
    case 0xC:
      return sub_E5ED80(a1, a2);
    case 0xD:
      return sub_E5EDD0((__int64)a1, a2);
    default:
      return 0;
  }
}
