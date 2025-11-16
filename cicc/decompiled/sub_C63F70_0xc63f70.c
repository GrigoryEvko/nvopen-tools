// Function: sub_C63F70
// Address: 0xc63f70
//
unsigned __int64 __fastcall sub_C63F70(
        unsigned __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-88h]
  _QWORD *v17; // [rsp+18h] [rbp-78h]
  __int64 *v18; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  __int64 v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h] BYREF
  __int64 v22; // [rsp+40h] [rbp-50h] BYREF
  __int64 v23; // [rsp+48h] [rbp-48h] BYREF
  __int64 v24; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v25[7]; // [rsp+58h] [rbp-38h] BYREF

  result = *a1 & 0xFFFFFFFFFFFFFFFELL;
  if ( result )
  {
    *a1 = result | 1;
    sub_CA0E80(&a7, a2);
    v8 = *a1;
    v18 = a2;
    *a1 = 0;
    v9 = v8 & 0xFFFFFFFFFFFFFFFELL;
    v19 = 0;
    v10 = (_QWORD *)v9;
    if ( v9 )
    {
      v20 = 0;
      a2 = (__int64 *)&unk_4F84052;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v9 + 48LL))(v9, &unk_4F84052) )
      {
        v11 = (_QWORD *)v10[2];
        v12 = (_QWORD *)v10[1];
        v21 = 1;
        v16 = v11;
        if ( v12 == v11 )
        {
          v15 = 1;
        }
        else
        {
          do
          {
            v17 = v12;
            v23 = *v12;
            *v12 = 0;
            sub_C63DA0(&v24, &v23, &v18);
            v13 = v21;
            a2 = &v22;
            v21 = 0;
            v22 = v13 | 1;
            sub_9CDB40(v25, (unsigned __int64 *)&v22, (unsigned __int64 *)&v24);
            if ( (v21 & 1) != 0 || (v14 = v17, (v21 & 0xFFFFFFFFFFFFFFFELL) != 0) )
              sub_C63C30(&v21, (__int64)&v22);
            v21 |= v25[0] | 1;
            if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v22, (__int64)&v22);
            if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v24, (__int64)&v22);
            if ( v23 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
              v14 = v17;
            }
            v12 = v14 + 1;
          }
          while ( v16 != v12 );
          v15 = v21 | 1;
        }
        v24 = v15;
        (*(void (__fastcall **)(_QWORD *))(*v10 + 8LL))(v10);
      }
      else
      {
        v25[0] = (unsigned __int64)v10;
        a2 = (__int64 *)v25;
        sub_C63DA0(&v24, v25, &v18);
        if ( v25[0] )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v25[0] + 8LL))(v25[0]);
      }
      if ( (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v20, (__int64)a2);
    }
    result = v19;
    if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v19, (__int64)a2);
  }
  else
  {
    *a1 = 0;
  }
  return result;
}
