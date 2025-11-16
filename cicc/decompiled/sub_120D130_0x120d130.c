// Function: sub_120D130
// Address: 0x120d130
//
__int64 __fastcall sub_120D130(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  int v4; // eax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-50h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  result = 0;
  *a2 = 2;
  if ( *(_DWORD *)(a1 + 240) == 12 )
  {
    v4 = sub_1205200(v2);
    *(_DWORD *)(a1 + 240) = v4;
    if ( v4 == 164 )
    {
      *a2 = 1;
      goto LABEL_5;
    }
    if ( v4 == 165 )
    {
      *a2 = 2;
LABEL_5:
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      return sub_120AFE0(a1, 13, "expected ')'");
    }
    v8 = 1;
    v5 = *(_QWORD *)(a1 + 232);
    v7 = 3;
    v6 = "expected unwind table kind";
    sub_11FD800(v2, v5, (__int64)&v6, 1);
    return 1;
  }
  return result;
}
