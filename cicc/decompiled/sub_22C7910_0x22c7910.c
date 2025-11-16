// Function: sub_22C7910
// Address: 0x22c7910
//
__int64 __fastcall sub_22C7910(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned int v6; // eax
  unsigned __int64 v7; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-98h]
  unsigned __int64 v9; // [rsp+10h] [rbp-90h]
  unsigned int v10; // [rsp+18h] [rbp-88h]
  unsigned __int64 v11; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-78h]
  unsigned __int64 v13; // [rsp+30h] [rbp-70h]
  unsigned int v14; // [rsp+38h] [rbp-68h]
  char v15; // [rsp+40h] [rbp-60h]
  unsigned __int8 v16[80]; // [rsp+50h] [rbp-50h] BYREF

  if ( (unsigned int)*a3 - 67 > 2 )
  {
    *(_BYTE *)(a1 + 40) = 1;
    *(_WORD *)a1 = 6;
    *(_WORD *)v16 = 0;
    sub_22C0090(v16);
    return a1;
  }
  else
  {
    sub_22C7770((__int64)&v11, a2, *((_QWORD *)a3 - 4), (__int64)a3, a4);
    if ( v15 )
    {
      v6 = sub_BCB060(*((_QWORD *)a3 + 1));
      sub_AB49F0((__int64)&v7, (__int64)&v11, *a3 - 29, v6);
      sub_22C06B0((__int64)v16, (__int64)&v7, 0);
      sub_22C0650(a1, v16);
      *(_BYTE *)(a1 + 40) = 1;
      sub_22C0090(v16);
      if ( v10 > 0x40 && v9 )
        j_j___libc_free_0_0(v9);
      if ( v8 > 0x40 && v7 )
        j_j___libc_free_0_0(v7);
      if ( v15 )
      {
        v15 = 0;
        if ( v14 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        if ( v12 > 0x40 )
        {
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
      }
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 0;
    }
    return a1;
  }
}
