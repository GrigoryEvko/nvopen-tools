// Function: sub_120AFE0
// Address: 0x120afe0
//
__int64 __fastcall sub_120AFE0(__int64 a1, int a2, _BYTE *a3)
{
  __int64 v4; // rdi
  bool v5; // zf
  _BYTE *v7; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+20h] [rbp-20h]

  v4 = a1 + 176;
  if ( *(_DWORD *)(a1 + 240) == a2 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v4);
    return 0;
  }
  else
  {
    v5 = *a3 == 0;
    v8 = 257;
    if ( !v5 )
    {
      v7 = a3;
      LOBYTE(v8) = 3;
    }
    sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)&v7, 1);
    return 1;
  }
}
