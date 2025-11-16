// Function: sub_38F09C0
// Address: 0x38f09c0
//
__int64 __fastcall sub_38F09C0(__int64 a1, char a2)
{
  char *v3; // rsi
  bool v4; // zf
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  const char *v8; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+10h] [rbp-30h]
  char v10; // [rsp+11h] [rbp-2Fh]

  v3 = *(char **)(a1 + 400);
  if ( v3 == *(char **)(a1 + 408) )
  {
    sub_38E9AD0((unsigned __int64 *)(a1 + 392), v3, (_QWORD *)(a1 + 380));
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)(a1 + 380);
      v3 = *(char **)(a1 + 400);
    }
    *(_QWORD *)(a1 + 400) = v3 + 8;
  }
  v4 = *(_BYTE *)(a1 + 385) == 0;
  *(_DWORD *)(a1 + 380) = 1;
  if ( v4 )
  {
    sub_38EAF10(a1);
    v10 = 1;
    v7 = v6;
    v8 = "unexpected token in '.ifb' directive";
    v9 = 3;
    result = sub_3909E20(a1, 9, &v8);
    if ( !(_BYTE)result )
    {
      *(_BYTE *)(a1 + 384) = a2 == (v7 == 0);
      *(_BYTE *)(a1 + 385) = a2 ^ (v7 == 0);
    }
  }
  else
  {
    sub_38F0630(a1);
    return 0;
  }
  return result;
}
