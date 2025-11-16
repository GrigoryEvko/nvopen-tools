// Function: sub_12BF260
// Address: 0x12bf260
//
__int64 *__fastcall sub_12BF260(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 *v3; // r15
  __int64 **v6; // rax
  __int64 **v7; // rcx
  __int64 **v8; // rcx
  __int64 v9; // rax
  __int64 **v10; // [rsp+0h] [rbp-80h]
  __int64 **v11; // [rsp+18h] [rbp-68h]
  unsigned __int64 v12; // [rsp+28h] [rbp-58h]
  __int64 *v13; // [rsp+30h] [rbp-50h] BYREF
  __int64 v14; // [rsp+38h] [rbp-48h] BYREF
  __int64 v15; // [rsp+40h] [rbp-40h] BYREF
  __int64 *v16[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = (__int64 *)(*a2 & 0xFFFFFFFFFFFFFFFELL);
  if ( v3 )
  {
    *a2 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, void *))(*v3 + 48))(v3, &unk_4FA032A) )
    {
      v6 = (__int64 **)v3[2];
      v7 = (__int64 **)v3[1];
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
          sub_12BF030(&v14, &v13, a3);
          v15 = v12 | 1;
          sub_12BEC00((unsigned __int64 *)v16, (unsigned __int64 *)&v15, (unsigned __int64 *)&v14);
          v8 = v11;
          v12 = (unsigned __int64)v16[0] | 1;
          if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v15);
          if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v14);
          if ( v13 )
          {
            (*(void (__fastcall **)(__int64 *))(*v13 + 8))(v13);
            v8 = v11;
          }
          v7 = v8 + 1;
        }
        while ( v10 != v7 );
        v9 = v12 | 1;
      }
      *a1 = v9;
      (*(void (__fastcall **)(__int64 *))(*v3 + 8))(v3);
    }
    else
    {
      v16[0] = v3;
      sub_12BF030(a1, v16, a3);
      if ( v16[0] )
        (*(void (__fastcall **)(__int64 *))(*v16[0] + 8))(v16[0]);
    }
  }
  else
  {
    *a2 = 0;
    *a1 = 1;
  }
  return a1;
}
