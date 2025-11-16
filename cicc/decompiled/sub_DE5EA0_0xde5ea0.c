// Function: sub_DE5EA0
// Address: 0xde5ea0
//
__int64 __fastcall sub_DE5EA0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 *v4; // r15
  unsigned int v5; // r14d
  char v6; // r12
  unsigned int v7; // eax
  unsigned int i; // ecx
  unsigned int v9; // eax
  __int64 *v11; // [rsp+8h] [rbp-88h]
  __int64 *v12; // [rsp+10h] [rbp-80h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  _BYTE v14[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = (unsigned __int64)&v12;
  v12 = (__int64 *)v14;
  v13 = 0x800000000LL;
  sub_D46D90(a2, (__int64)&v12);
  v4 = v12;
  v11 = &v12[(unsigned int)v13];
  if ( v12 != v11 )
  {
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      v7 = sub_DE5E70(a1, a2, *v4);
      if ( !v6 )
        v5 = v7;
      v3 = v7;
      if ( !v5 )
        goto LABEL_10;
      if ( v7 )
        break;
LABEL_11:
      ++v4;
      v6 = 1;
      if ( v11 == v4 )
      {
        v11 = v12;
        goto LABEL_13;
      }
    }
    for ( i = v5 % v7; i; i = v9 % i )
    {
      v9 = v3;
      v3 = i;
    }
LABEL_10:
    v5 = v3;
    goto LABEL_11;
  }
  v5 = 1;
LABEL_13:
  if ( v11 != (__int64 *)v14 )
    _libc_free(v11, v3);
  return v5;
}
