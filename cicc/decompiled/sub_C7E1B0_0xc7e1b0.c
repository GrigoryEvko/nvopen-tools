// Function: sub_C7E1B0
// Address: 0xc7e1b0
//
__int64 __fastcall sub_C7E1B0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  bool v3; // zf
  unsigned __int64 v4; // rax
  _QWORD *v5; // rbx
  __int64 *v6; // rax
  __int64 *v7; // rcx
  __int64 *v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // [rsp+8h] [rbp-98h]
  __int64 *v11; // [rsp+18h] [rbp-88h]
  __int64 v12; // [rsp+20h] [rbp-80h] BYREF
  __int64 v13; // [rsp+28h] [rbp-78h] BYREF
  __int64 v14; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v15; // [rsp+38h] [rbp-68h]
  unsigned __int64 v16; // [rsp+40h] [rbp-60h] BYREF
  __int64 v17; // [rsp+48h] [rbp-58h] BYREF
  __int64 v18; // [rsp+50h] [rbp-50h] BYREF
  __int64 v19; // [rsp+58h] [rbp-48h] BYREF
  __int64 v20; // [rsp+60h] [rbp-40h] BYREF
  char v21; // [rsp+68h] [rbp-38h]

  sub_C85FC0(&v20);
  if ( (v21 & 1) == 0 )
    return (unsigned int)v20;
  v12 = 0;
  v21 &= ~2u;
  v13 = 0;
  v4 = v20 & 0xFFFFFFFFFFFFFFFELL;
  v3 = (v20 & 0xFFFFFFFFFFFFFFFELL) == 0;
  v20 = 0;
  v5 = (_QWORD *)v4;
  if ( !v3 )
  {
    v14 = 0;
    a2 = (__int64 *)&unk_4F84052;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v4 + 48LL))(v4, &unk_4F84052) )
    {
      v6 = (__int64 *)v5[2];
      v7 = (__int64 *)v5[1];
      v15 = 1;
      v10 = v6;
      if ( v7 == v6 )
      {
        v9 = 1;
      }
      else
      {
        do
        {
          v11 = v7;
          v18 = *v7;
          *v7 = 0;
          sub_C7D900(&v17, &v18);
          a2 = &v19;
          v19 = v15 | 1;
          sub_9CDB40(&v16, (unsigned __int64 *)&v19, (unsigned __int64 *)&v17);
          v8 = v11;
          v15 = v16 | 1;
          if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v19, (__int64)&v19);
          if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v17, (__int64)&v19);
          if ( v18 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
            v8 = v11;
          }
          v7 = v8 + 1;
        }
        while ( v10 != v7 );
        v9 = v15 | 1;
      }
      v18 = v9;
      (*(void (__fastcall **)(_QWORD *))(*v5 + 8LL))(v5);
    }
    else
    {
      a2 = &v19;
      v19 = (__int64)v5;
      sub_C7D900(&v18, &v19);
      if ( v19 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
    }
    if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v14, (__int64)a2);
  }
  if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v13, (__int64)a2);
  if ( (v12 & 1) != 0 || (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v12, (__int64)a2);
  if ( (v21 & 2) != 0 )
    sub_9CE230(&v20);
  if ( (v21 & 1) == 0 )
    return 4096;
  result = 4096;
  if ( v20 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
    return 4096;
  }
  return result;
}
