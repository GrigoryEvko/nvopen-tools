// Function: sub_24AC500
// Address: 0x24ac500
//
__int64 __fastcall sub_24AC500(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 result; // rax
  __int64 *v6; // rbx
  __int64 **v7; // rax
  __int64 **v8; // rcx
  __int64 v9; // rax
  __int64 **v10; // rcx
  __int64 v11; // rax
  __int64 **v12; // [rsp+0h] [rbp-A0h]
  __int64 **v13; // [rsp+10h] [rbp-90h]
  __int64 v14; // [rsp+18h] [rbp-88h] BYREF
  __int64 v15; // [rsp+28h] [rbp-78h] BYREF
  __int64 v16; // [rsp+30h] [rbp-70h] BYREF
  __int64 v17; // [rsp+38h] [rbp-68h] BYREF
  __int64 v18; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v19; // [rsp+48h] [rbp-58h] BYREF
  __int64 v20; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v21; // [rsp+58h] [rbp-48h] BYREF
  __int64 *v22[8]; // [rsp+60h] [rbp-40h] BYREF

  v22[1] = &v14;
  v3 = *a2;
  v14 = a3;
  v22[0] = a1;
  *a2 = 0;
  v15 = 0;
  v4 = v3 & 0xFFFFFFFFFFFFFFFELL;
  if ( v4 )
  {
    v6 = (__int64 *)v4;
    v16 = 0;
    a2 = (__int64 *)&unk_4F84052;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v4 + 48LL))(v4, &unk_4F84052) )
    {
      v7 = (__int64 **)v6[2];
      v8 = (__int64 **)v6[1];
      v17 = 1;
      v12 = v7;
      if ( v8 == v7 )
      {
        v11 = 1;
      }
      else
      {
        do
        {
          v13 = v8;
          v19 = *v8;
          *v8 = 0;
          sub_24AAAB0(&v20, &v19, v22);
          v9 = v17;
          a2 = &v18;
          v17 = 0;
          v18 = v9 | 1;
          sub_9CDB40(&v21, (unsigned __int64 *)&v18, (unsigned __int64 *)&v20);
          if ( (v17 & 1) != 0 || (v10 = v13, (v17 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v17, (__int64)&v18);
          v17 |= v21 | 1;
          if ( (v18 & 1) != 0 || (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v18, (__int64)&v18);
          if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v20, (__int64)&v18);
          if ( v19 )
          {
            (*(void (__fastcall **)(__int64 *))(*v19 + 8))(v19);
            v10 = v13;
          }
          v8 = v10 + 1;
        }
        while ( v12 != v8 );
        v11 = v17 | 1;
      }
      v20 = v11;
      (*(void (__fastcall **)(__int64 *))(*v6 + 8))(v6);
    }
    else
    {
      v21 = (unsigned __int64)v6;
      a2 = (__int64 *)&v21;
      sub_24AAAB0(&v20, (__int64 **)&v21, v22);
      if ( v21 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v21 + 8LL))(v21);
    }
    if ( (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v16 & 1) != 0 || (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v16, (__int64)a2);
  }
  result = v15;
  if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v15, (__int64)a2);
  return result;
}
