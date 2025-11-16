// Function: sub_1243920
// Address: 0x1243920
//
__int64 __fastcall sub_1243920(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 result; // rax
  unsigned int v3; // eax
  unsigned __int64 v4; // rsi
  const char *v5; // [rsp+0h] [rbp-50h] BYREF
  char v6; // [rsp+20h] [rbp-30h]
  char v7; // [rsp+21h] [rbp-2Fh]

  *(_BYTE *)(a1 + 336) = 1;
  v1 = *(_DWORD *)(a1 + 280);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  result = sub_120AFE0(a1, 3, "expected '=' here");
  if ( !(_BYTE)result )
  {
    if ( !*(_QWORD *)(a1 + 352) )
      return sub_1210D90(a1);
    v3 = *(_DWORD *)(a1 + 240);
    if ( v3 == 416 )
    {
      result = sub_1210D20(a1);
      goto LABEL_11;
    }
    if ( v3 > 0x1A0 )
    {
      if ( v3 == 461 )
      {
        result = sub_123C800(a1, v1);
        goto LABEL_11;
      }
      if ( v3 == 462 )
      {
        result = sub_1239690(a1, v1);
        goto LABEL_11;
      }
    }
    else
    {
      switch ( v3 )
      {
        case 0x19Bu:
          result = sub_12434B0(a1, v1);
          goto LABEL_11;
        case 0x19Fu:
          result = sub_1210CA0(a1);
LABEL_11:
          *(_BYTE *)(a1 + 336) = 0;
          return result;
        case 0x64u:
          result = sub_1238C00(a1, v1);
          goto LABEL_11;
      }
    }
    v7 = 1;
    v4 = *(_QWORD *)(a1 + 232);
    v6 = 3;
    v5 = "unexpected summary kind";
    sub_11FD800(a1 + 176, v4, (__int64)&v5, 1);
    result = 1;
    goto LABEL_11;
  }
  return result;
}
