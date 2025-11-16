// Function: sub_393B110
// Address: 0x393b110
//
__int64 __fastcall sub_393B110(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v6; // rdx
  __int64 v7; // [rsp+8h] [rbp-28h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF
  char v9; // [rsp+18h] [rbp-18h]

  sub_39388E0((__int64)&v8, a2);
  v2 = (v9 & 1) == 0;
  v3 = v8;
  v9 &= ~2u;
  if ( !v2 )
  {
    v4 = v8 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v4;
      return a1;
    }
    v3 = 0;
  }
  v7 = v3;
  v8 = 0;
  sub_3939480(a1, &v7);
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  if ( (v9 & 2) != 0 )
    sub_1880CF0(&v8, (__int64)&v7, v6);
  if ( !v8 )
    return a1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  return a1;
}
