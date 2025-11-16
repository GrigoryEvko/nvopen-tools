// Function: sub_22CCDF0
// Address: 0x22ccdf0
//
__int64 __fastcall sub_22CCDF0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6)
{
  unsigned __int8 v7; // al
  int v8; // edx
  __int64 v10; // [rsp+8h] [rbp-98h]
  __int64 v11; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-88h]
  __int64 v13[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v14[10]; // [rsp+50h] [rbp-50h] BYREF

  v7 = *a3;
  if ( *a3 > 0x15u )
  {
    LOWORD(v11) = 6;
    if ( v7 > 0x1Cu )
    {
      v10 = a4;
      sub_22C07D0(v13, a3);
      sub_22C0090((unsigned __int8 *)&v11);
      sub_22C0650((__int64)&v11, (unsigned __int8 *)v13);
      sub_22C0090((unsigned __int8 *)v13);
      a4 = v10;
    }
    sub_22C6BD0(a2, (__int64)a3, (unsigned __int8 *)&v11, a4, a5, a6);
    sub_22C0650(a1, (unsigned __int8 *)&v11);
    sub_22C0090((unsigned __int8 *)&v11);
  }
  else
  {
    *(_WORD *)a1 = 0;
    v8 = *a3;
    if ( (unsigned int)(v8 - 12) > 1 )
    {
      if ( (_BYTE)v8 == 17 )
      {
        v12 = *((_DWORD *)a3 + 8);
        if ( v12 > 0x40 )
          sub_C43780((__int64)&v11, (const void **)a3 + 3);
        else
          v11 = *((_QWORD *)a3 + 3);
        sub_AADBC0((__int64)v13, &v11);
        sub_22C00F0(a1, (__int64)v13, 0, 0, 1u);
        sub_969240(v14);
        sub_969240(v13);
        sub_969240(&v11);
      }
      else
      {
        *(_BYTE *)a1 = 2;
        *(_QWORD *)(a1 + 8) = a3;
      }
    }
    else
    {
      *(_BYTE *)a1 = 1;
    }
  }
  return a1;
}
