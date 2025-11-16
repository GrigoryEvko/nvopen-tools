// Function: sub_EE20B0
// Address: 0xee20b0
//
_WORD *__fastcall sub_EE20B0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rsi
  _WORD *v6; // rdi
  __int64 v7; // rcx
  char v9; // [rsp+Fh] [rbp-41h] BYREF
  _WORD *v10; // [rsp+10h] [rbp-40h]
  unsigned __int64 v11; // [rsp+18h] [rbp-38h]
  _WORD *v12; // [rsp+20h] [rbp-30h] BYREF
  unsigned __int64 v13; // [rsp+28h] [rbp-28h]

  v10 = 0;
  v11 = 0;
  v12 = (_WORD *)a1;
  v13 = a2;
  do
  {
    v9 = 59;
    v3 = sub_C931B0((__int64 *)&v12, &v9, 1u, 0);
    if ( v3 == -1 )
    {
      v6 = v12;
      v3 = v13;
      v5 = 0;
      v7 = 0;
    }
    else
    {
      v4 = v3 + 1;
      if ( v3 + 1 > v13 )
      {
        v4 = v13;
        v5 = 0;
      }
      else
      {
        v5 = v13 - v4;
      }
      v6 = v12;
      v7 = (__int64)v12 + v4;
      if ( v3 > v13 )
        v3 = v13;
    }
    v10 = v6;
    v11 = v3;
    v12 = (_WORD *)v7;
    v13 = v5;
    if ( v3 > 1 && *v6 == 23135 )
      return v6;
  }
  while ( v5 );
  return (_WORD *)a1;
}
