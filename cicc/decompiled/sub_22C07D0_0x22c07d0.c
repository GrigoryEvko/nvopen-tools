// Function: sub_22C07D0
// Address: 0x22c07d0
//
_WORD *__fastcall sub_22C07D0(_WORD *a1, unsigned __int8 *a2)
{
  int v3; // edx
  __int64 v5; // rsi
  __int64 v6; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-78h]
  __int64 v8; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-68h]
  __int64 v10; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-58h]
  __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-48h]
  char v14; // [rsp+40h] [rbp-40h]

  v3 = *a2;
  if ( v3 == 61 )
    goto LABEL_9;
  if ( v3 != 85 && v3 != 34 )
  {
LABEL_4:
    *a1 = 6;
    return a1;
  }
  sub_B492D0((__int64)&v10, (__int64)a2);
  if ( !v14 )
  {
LABEL_9:
    if ( (a2[7] & 0x20) != 0 )
    {
      v5 = sub_B91C10((__int64)a2, 4);
      if ( v5 )
      {
        if ( *(_BYTE *)(*((_QWORD *)a2 + 1) + 8LL) == 12 )
        {
          sub_ABEA30((__int64)&v10, v5);
          sub_22C06B0((__int64)a1, (__int64)&v10, 0);
          sub_969240(&v12);
          sub_969240(&v10);
          return a1;
        }
      }
    }
    goto LABEL_4;
  }
  v7 = v11;
  if ( v11 > 0x40 )
    sub_C43780((__int64)&v6, (const void **)&v10);
  else
    v6 = v10;
  v9 = v13;
  if ( v13 > 0x40 )
    sub_C43780((__int64)&v8, (const void **)&v12);
  else
    v8 = v12;
  sub_22C06B0((__int64)a1, (__int64)&v6, 0);
  sub_969240(&v8);
  sub_969240(&v6);
  if ( v14 )
  {
    v14 = 0;
    sub_969240(&v12);
    sub_969240(&v10);
  }
  return a1;
}
