// Function: sub_388AF10
// Address: 0x388af10
//
__int64 __fastcall sub_388AF10(__int64 a1, int a2, _BYTE *a3)
{
  __int64 v4; // rdi
  bool v5; // zf
  _BYTE *v7; // [rsp+0h] [rbp-30h] BYREF
  __int16 v8; // [rsp+10h] [rbp-20h]

  v4 = a1 + 8;
  if ( *(_DWORD *)(a1 + 64) == a2 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
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
    return sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v7);
  }
}
