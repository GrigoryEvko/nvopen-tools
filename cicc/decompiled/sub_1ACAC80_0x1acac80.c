// Function: sub_1ACAC80
// Address: 0x1acac80
//
__int64 __fastcall sub_1ACAC80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ebx
  unsigned int v4; // eax
  int v6; // eax
  __int64 *v7; // r12
  __int64 v8; // r13
  __int64 *v9; // rbx
  int v10; // [rsp+Ch] [rbp-94h]
  int v11; // [rsp+30h] [rbp-70h]
  unsigned int v12; // [rsp+34h] [rbp-6Ch]
  __int64 v13; // [rsp+38h] [rbp-68h]
  __int64 v14; // [rsp+40h] [rbp-60h] BYREF
  __int64 v15; // [rsp+48h] [rbp-58h] BYREF
  __int64 v16; // [rsp+50h] [rbp-50h] BYREF
  __int64 v17; // [rsp+58h] [rbp-48h] BYREF
  __int64 v18; // [rsp+60h] [rbp-40h] BYREF
  __int64 v19[7]; // [rsp+68h] [rbp-38h] BYREF

  v15 = a2;
  v14 = a3;
  v3 = sub_15601D0((__int64)&v14);
  v4 = sub_15601D0((__int64)&v15);
  v12 = sub_1ACA9E0(a1, v4, v3);
  if ( !v12 )
  {
    v6 = sub_15601D0((__int64)&v15);
    v11 = -1;
    v10 = v6 - 1;
    if ( v6 )
    {
      while ( 1 )
      {
        v16 = sub_15601E0(&v15, v11);
        v17 = sub_15601E0(&v14, v11);
        v7 = (__int64 *)sub_155EE30(&v16);
        v8 = sub_155EE40(&v16);
        v9 = (__int64 *)sub_155EE30(&v17);
        v13 = sub_155EE40(&v17);
        if ( v9 != (__int64 *)v13 && v7 != (__int64 *)v8 )
        {
          do
          {
            v18 = *v7;
            v19[0] = *v9;
            if ( sub_155E9A0(&v18, v19[0]) )
              return (unsigned int)-1;
            if ( sub_155E9A0(v19, v18) )
              return 1;
            ++v7;
            ++v9;
            if ( (__int64 *)v8 == v7 )
              goto LABEL_13;
          }
          while ( (__int64 *)v13 != v9 );
        }
        if ( (__int64 *)v8 != v7 )
          return 1;
LABEL_13:
        if ( (__int64 *)v13 != v9 )
          return (unsigned int)-1;
        if ( ++v11 == v10 )
          return v12;
      }
    }
  }
  return v12;
}
