// Function: sub_1C2FF50
// Address: 0x1c2ff50
//
__int64 __fastcall sub_1C2FF50(__int64 a1, int a2, _DWORD *a3)
{
  unsigned int v4; // r13d
  _BYTE *v5; // rdi
  _DWORD *v6; // rax
  _BYTE *v8; // [rsp+0h] [rbp-70h] BYREF
  __int64 v9; // [rsp+8h] [rbp-68h]
  _BYTE v10[96]; // [rsp+10h] [rbp-60h] BYREF

  v8 = v10;
  v9 = 0x1000000000LL;
  v4 = sub_1C2E2E0(a1, "align", 5u, (__int64)&v8);
  if ( (_BYTE)v4 )
  {
    v5 = v8;
    if ( (int)v9 <= 0 )
    {
LABEL_11:
      v4 = 0;
    }
    else
    {
      v6 = v8;
      while ( HIWORD(*v6) != a2 )
      {
        if ( &v8[4 * (unsigned int)(v9 - 1) + 4] == (_BYTE *)++v6 )
          goto LABEL_11;
      }
      *a3 = (unsigned __int16)*v6;
    }
  }
  else
  {
    v5 = v8;
  }
  if ( v5 != v10 )
    _libc_free((unsigned __int64)v5);
  return v4;
}
