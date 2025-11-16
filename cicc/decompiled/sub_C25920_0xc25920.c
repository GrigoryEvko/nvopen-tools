// Function: sub_C25920
// Address: 0xc25920
//
__int64 *__fastcall sub_C25920(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  _QWORD *v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // [rsp+8h] [rbp-78h]
  _QWORD *v10; // [rsp+18h] [rbp-68h]
  __int64 v11; // [rsp+28h] [rbp-58h] BYREF
  __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  __int64 v13; // [rsp+38h] [rbp-48h] BYREF
  __int64 v14; // [rsp+40h] [rbp-40h] BYREF
  __int64 v15[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = (_QWORD *)(v2 & 0xFFFFFFFFFFFFFFFELL);
  if ( v3 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v3 + 48LL))(v3, &unk_4F84052) )
    {
      v5 = (_QWORD *)v3[2];
      v6 = (_QWORD *)v3[1];
      v11 = 1;
      v9 = v5;
      if ( v6 == v5 )
      {
        v8 = 1;
      }
      else
      {
        do
        {
          v10 = v6;
          v13 = *v6;
          *v6 = 0;
          sub_C1F740(&v14, &v13);
          v12 = v11 | 1;
          sub_9CDB40((unsigned __int64 *)v15, (unsigned __int64 *)&v12, (unsigned __int64 *)&v14);
          v7 = v10;
          v11 = v15[0] | 1;
          if ( (v12 & 1) != 0 || (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v12);
          if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v14);
          if ( v13 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
            v7 = v10;
          }
          v6 = v7 + 1;
        }
        while ( v9 != v6 );
        v8 = v11 | 1;
      }
      *a1 = v8;
      v11 = 0;
      sub_9C66B0(&v11);
      (*(void (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
    }
    else
    {
      v15[0] = (__int64)v3;
      sub_C1F740(a1, v15);
      if ( v15[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v15[0] + 8LL))(v15[0]);
    }
  }
  else
  {
    *a1 = 1;
    v15[0] = 0;
    sub_9C66B0(v15);
  }
  return a1;
}
