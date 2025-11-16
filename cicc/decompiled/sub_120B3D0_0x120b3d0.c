// Function: sub_120B3D0
// Address: 0x120b3d0
//
__int64 __fastcall sub_120B3D0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 240) == 512 )
  {
    sub_2240AE0(a2, a1 + 248);
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 232);
    v6 = 1;
    v5 = 3;
    v4 = "expected string constant";
    sub_11FD800(a1 + 176, v2, (__int64)&v4, 1);
    return 1;
  }
}
