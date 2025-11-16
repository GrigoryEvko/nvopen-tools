// Function: sub_120A860
// Address: 0x120a860
//
__int64 __fastcall sub_120A860(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r12d
  unsigned int v4; // eax
  unsigned int v5; // r12d
  unsigned __int64 v7; // rsi
  const char *v8; // [rsp+0h] [rbp-50h] BYREF
  char v9; // [rsp+20h] [rbp-30h]
  char v10; // [rsp+21h] [rbp-2Fh]

  v3 = 0;
  if ( (*(_BYTE *)(a2 + 34) & 1) != 0 )
    LOBYTE(v3) = *(_BYTE *)sub_B31490(a2, a2, a3);
  v4 = *(_DWORD *)(a1 + 240);
  if ( v4 == 500 )
  {
    v5 = v3 | 2;
    goto LABEL_8;
  }
  if ( v4 > 0x1F4 )
  {
    v5 = v3 | 8;
    if ( v4 == 501 )
      goto LABEL_8;
  }
  else
  {
    if ( v4 == 223 )
    {
      v5 = v3 | 4;
      goto LABEL_8;
    }
    if ( v4 == 499 )
    {
      v5 = v3 | 1;
LABEL_8:
      sub_B311F0(a2, v5, a3);
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      return 0;
    }
  }
  v10 = 1;
  v7 = *(_QWORD *)(a1 + 232);
  v9 = 3;
  v8 = "non-sanitizer token passed to LLParser::parseSanitizer()";
  sub_11FD800(a1 + 176, v7, (__int64)&v8, 1);
  return 1;
}
