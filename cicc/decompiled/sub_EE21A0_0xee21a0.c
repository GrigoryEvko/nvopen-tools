// Function: sub_EE21A0
// Address: 0xee21a0
//
unsigned __int64 *__fastcall sub_EE21A0(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rbx
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // rcx
  unsigned __int64 *v7; // rcx
  __int64 v8; // rax
  unsigned __int64 *v9; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v10; // [rsp+18h] [rbp-68h]
  unsigned __int64 v11; // [rsp+28h] [rbp-58h]
  unsigned __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v13; // [rsp+38h] [rbp-48h] BYREF
  __int64 v14; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v15[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = (_QWORD *)(v2 & 0xFFFFFFFFFFFFFFFELL);
  if ( v3 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v3 + 48LL))(v3, &unk_4F84052) )
    {
      v5 = (unsigned __int64 *)v3[2];
      v6 = (unsigned __int64 *)v3[1];
      v11 = 1;
      v9 = v5;
      if ( v5 == v6 )
      {
        v8 = 1;
      }
      else
      {
        do
        {
          v10 = v6;
          v12 = *v6;
          *v6 = 0;
          sub_ED8110(&v13, &v12);
          v14 = v11 | 1;
          sub_9CDB40(v15, (unsigned __int64 *)&v14, &v13);
          v7 = v10;
          v11 = v15[0] | 1;
          if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v14, (__int64)&v14);
          if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v13, (__int64)&v14);
          if ( v12 )
          {
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v12 + 8LL))(v12);
            v7 = v10;
          }
          v6 = v7 + 1;
        }
        while ( v9 != v6 );
        v8 = v11 | 1;
      }
      *a1 = v8;
      (*(void (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
    }
    else
    {
      v15[0] = (unsigned __int64)v3;
      sub_ED8110(a1, v15);
      if ( v15[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v15[0] + 8LL))(v15[0]);
    }
  }
  else
  {
    *a1 = 1;
  }
  return a1;
}
