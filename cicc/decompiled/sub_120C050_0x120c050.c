// Function: sub_120C050
// Address: 0x120c050
//
__int64 __fastcall sub_120C050(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rsi
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned int v6; // r12d
  const char *v7; // [rsp+10h] [rbp-40h] BYREF
  char v8; // [rsp+30h] [rbp-20h]
  char v9; // [rsp+31h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 240) == 529 && *(_BYTE *)(a1 + 332) )
  {
    v4 = *(_DWORD *)(a1 + 328);
    if ( v4 > 0x40 )
    {
      v6 = v4 - sub_C444A0(a1 + 320);
      v5 = -1;
      if ( v6 <= 0x40 )
        v5 = **(_QWORD **)(a1 + 320);
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 320);
    }
    *a2 = v5;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 232);
    v9 = 1;
    v7 = "expected integer";
    v8 = 3;
    sub_11FD800(a1 + 176, v2, (__int64)&v7, 1);
    return 1;
  }
}
