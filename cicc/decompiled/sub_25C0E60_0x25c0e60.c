// Function: sub_25C0E60
// Address: 0x25c0e60
//
__int64 __fastcall sub_25C0E60(__int64 a1)
{
  int v1; // eax
  unsigned int v2; // r12d
  int v4; // r13d
  __int64 v5; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v6; // [rsp+8h] [rbp-68h]
  __int64 v7; // [rsp+10h] [rbp-60h]
  int v8; // [rsp+18h] [rbp-58h]
  char v9; // [rsp+1Ch] [rbp-54h]
  __int64 v10; // [rsp+20h] [rbp-50h] BYREF

  v6 = &v10;
  v7 = 0x100000008LL;
  v8 = 0;
  v9 = 1;
  v10 = a1;
  v5 = 1;
  v1 = sub_25C0650(a1, (__int64)&v5);
  if ( v1 )
  {
    v4 = v1;
    if ( !(unsigned __int8)sub_B2D670(a1, v1) )
    {
      sub_B2D5C0(a1, 78);
      sub_B2D5C0(a1, 51);
      sub_B2D5C0(a1, 50);
      if ( (unsigned int)(v4 - 50) <= 1 )
        sub_B2D5C0(a1, 77);
      v2 = 1;
      sub_B2D400(a1, v4);
      if ( !v9 )
        goto LABEL_8;
      return v2;
    }
  }
  v2 = 0;
  if ( v9 )
    return v2;
LABEL_8:
  _libc_free((unsigned __int64)v6);
  return v2;
}
