// Function: sub_123C000
// Address: 0x123c000
//
__int64 __fastcall sub_123C000(__int64 a1, __int64 a2)
{
  int v3; // eax
  const char *v4; // rax
  unsigned __int64 v5; // rsi
  int v6; // eax
  int v7; // eax
  const char *v8; // [rsp+0h] [rbp-50h] BYREF
  char v9; // [rsp+20h] [rbp-30h]
  char v10; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int8)sub_120AFE0(a1, 478, "expected 'wpdRes' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_120AFE0(a1, 465, "expected 'kind' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
  {
    return 1;
  }
  v3 = *(_DWORD *)(a1 + 240);
  switch ( v3 )
  {
    case 480:
      *(_DWORD *)a2 = 1;
      break;
    case 481:
      *(_DWORD *)a2 = 2;
      break;
    case 479:
      *(_DWORD *)a2 = 0;
      break;
    default:
      v10 = 1;
      v4 = "unexpected WholeProgramDevirtResolution kind";
      goto LABEL_11;
  }
  v6 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v6;
  if ( v6 != 4 )
    return sub_120AFE0(a1, 13, "expected ')' here");
  while ( 1 )
  {
    v7 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v7;
    if ( v7 != 482 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") || (unsigned __int8)sub_120B3D0(a1, a2 + 8) )
      return 1;
LABEL_16:
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return sub_120AFE0(a1, 13, "expected ')' here");
  }
  if ( v7 == 483 )
  {
    if ( (unsigned __int8)sub_123B840(a1, (_QWORD *)(a2 + 40)) )
      return 1;
    goto LABEL_16;
  }
  v10 = 1;
  v4 = "expected optional WholeProgramDevirtResolution field";
LABEL_11:
  v8 = v4;
  v5 = *(_QWORD *)(a1 + 232);
  v9 = 3;
  sub_11FD800(a1 + 176, v5, (__int64)&v8, 1);
  return 1;
}
