// Function: sub_1254BC0
// Address: 0x1254bc0
//
__int64 __fastcall sub_1254BC0(__int64 a1)
{
  __int64 *v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  _QWORD *v6; // rbx
  __int64 *v7; // rax
  __int64 *v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // [rsp+8h] [rbp-A8h]
  __int64 *v13; // [rsp+18h] [rbp-98h]
  __int64 v14; // [rsp+28h] [rbp-88h] BYREF
  __int64 v15; // [rsp+30h] [rbp-80h] BYREF
  __int64 v16; // [rsp+38h] [rbp-78h] BYREF
  __int64 v17; // [rsp+40h] [rbp-70h] BYREF
  __int64 v18; // [rsp+48h] [rbp-68h] BYREF
  unsigned __int64 v19; // [rsp+50h] [rbp-60h] BYREF
  __int64 v20; // [rsp+58h] [rbp-58h] BYREF
  __int64 v21; // [rsp+60h] [rbp-50h] BYREF
  __int64 v22; // [rsp+68h] [rbp-48h] BYREF
  _QWORD v23[8]; // [rsp+70h] [rbp-40h] BYREF

  v1 = (__int64 *)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 56);
  v23[0] = 0;
  v23[1] = 0;
  sub_1255430(&v14, a1 + 8, v2, 1, v23);
  v3 = v14;
  v15 = 0;
  v14 = 0;
  v16 = 0;
  v4 = v3 & 0xFFFFFFFFFFFFFFFELL;
  if ( v4 )
  {
    v6 = (_QWORD *)v4;
    v17 = 0;
    v1 = (__int64 *)&unk_4F84052;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v4 + 48LL))(v4, &unk_4F84052) )
    {
      v7 = (__int64 *)v6[2];
      v8 = (__int64 *)v6[1];
      v18 = 1;
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
          v21 = *v8;
          *v8 = 0;
          sub_1254830(&v20, &v21);
          v9 = v18;
          v1 = &v22;
          v18 = 0;
          v22 = v9 | 1;
          sub_9CDB40(&v19, (unsigned __int64 *)&v22, (unsigned __int64 *)&v20);
          if ( (v18 & 1) != 0 || (v10 = v13, (v18 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v18, (__int64)&v22);
          v18 |= v19 | 1;
          if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v22, (__int64)&v22);
          if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v20, (__int64)&v22);
          if ( v21 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
            v10 = v13;
          }
          v8 = v10 + 1;
        }
        while ( v12 != v8 );
        v11 = v18 | 1;
      }
      v21 = v11;
      (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
    }
    else
    {
      v1 = &v22;
      v22 = (__int64)v6;
      sub_1254830(&v21, &v22);
      if ( v22 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 8LL))(v22);
    }
    if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v17, (__int64)v1);
  }
  if ( (v16 & 1) != 0 || (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v16, (__int64)v1);
  if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v15, (__int64)v1);
  if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v14, (__int64)v1);
  return *(unsigned __int8 *)v23[0];
}
