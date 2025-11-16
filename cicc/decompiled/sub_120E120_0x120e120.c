// Function: sub_120E120
// Address: 0x120e120
//
__int64 __fastcall sub_120E120(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 v3; // r15
  int v6; // eax
  unsigned __int64 v7; // rsi
  const char *v8; // rax
  unsigned int v9; // r14d
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  v3 = a1 + 176;
  v6 = sub_1205200(a1 + 176);
  v7 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = v6;
  if ( v6 == 12 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v3);
    v9 = sub_120BD00(a1, a2);
    if ( (_BYTE)v9 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) == 4 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v3);
      if ( (unsigned __int8)sub_120BD00(a1, a3) )
        return 1;
    }
    else
    {
      *a3 = *a2;
    }
    v7 = *(_QWORD *)(a1 + 232);
    if ( *(_DWORD *)(a1 + 240) == 13 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v3);
      return v9;
    }
    v13 = 1;
    v8 = "expected ')'";
  }
  else
  {
    v13 = 1;
    v8 = "expected '('";
  }
  v11 = v8;
  v12 = 3;
  sub_11FD800(v3, v7, (__int64)&v11, 1);
  return 1;
}
