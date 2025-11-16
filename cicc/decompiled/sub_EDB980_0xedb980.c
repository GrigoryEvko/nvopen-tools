// Function: sub_EDB980
// Address: 0xedb980
//
__int64 *__fastcall sub_EDB980(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // r15
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rcx
  unsigned __int64 *v8; // rcx
  __int64 v9; // rax
  unsigned __int64 *v10; // [rsp+0h] [rbp-80h]
  unsigned __int64 *v11; // [rsp+18h] [rbp-68h]
  unsigned __int64 v12; // [rsp+28h] [rbp-58h]
  unsigned __int64 v13; // [rsp+30h] [rbp-50h] BYREF
  __int64 v14; // [rsp+38h] [rbp-48h] BYREF
  __int64 v15; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v16[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = (_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFFELL);
  if ( v3 )
  {
    *a2 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v3 + 48LL))(v3, &unk_4F84052) )
    {
      v6 = (unsigned __int64 *)v3[2];
      v7 = (unsigned __int64 *)v3[1];
      v12 = 1;
      v10 = v6;
      if ( v6 == v7 )
      {
        v9 = 1;
      }
      else
      {
        do
        {
          v11 = v7;
          v13 = *v7;
          *v7 = 0;
          sub_EDB8F0(&v14, &v13, a3);
          v15 = v12 | 1;
          sub_9CDB40(v16, (unsigned __int64 *)&v15, (unsigned __int64 *)&v14);
          v8 = v11;
          v12 = v16[0] | 1;
          if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v15, (__int64)&v15);
          if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v14, (__int64)&v15);
          if ( v13 )
          {
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v13 + 8LL))(v13);
            v8 = v11;
          }
          v7 = v8 + 1;
        }
        while ( v10 != v7 );
        v9 = v12 | 1;
      }
      *a1 = v9;
      (*(void (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
    }
    else
    {
      v16[0] = (unsigned __int64)v3;
      sub_EDB8F0(a1, v16, a3);
      if ( v16[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v16[0] + 8LL))(v16[0]);
    }
  }
  else
  {
    *a2 = 0;
    *a1 = 1;
  }
  return a1;
}
