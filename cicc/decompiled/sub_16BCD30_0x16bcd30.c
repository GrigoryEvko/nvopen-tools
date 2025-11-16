// Function: sub_16BCD30
// Address: 0x16bcd30
//
unsigned __int64 __fastcall sub_16BCD30(
        unsigned __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // [rsp+8h] [rbp-88h]
  _QWORD *v19; // [rsp+18h] [rbp-78h]
  __int64 *v20; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-68h] BYREF
  __int64 v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h] BYREF
  __int64 v24; // [rsp+40h] [rbp-50h] BYREF
  __int64 v25; // [rsp+48h] [rbp-48h] BYREF
  unsigned __int64 v26; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v27[7]; // [rsp+58h] [rbp-38h] BYREF

  result = *a1 & 0xFFFFFFFFFFFFFFFELL;
  if ( result )
  {
    *a1 = result | 1;
    sub_16E2CE0(&a7, a2);
    v9 = *a1;
    v20 = a2;
    *a1 = 0;
    v10 = v9 & 0xFFFFFFFFFFFFFFFELL;
    v21 = 0;
    v11 = (_QWORD *)v10;
    if ( v10 )
    {
      v22 = 0;
      a2 = (__int64 *)&unk_4FA032A;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v10 + 48LL))(v10, &unk_4FA032A) )
      {
        v12 = (_QWORD *)v11[2];
        v13 = (_QWORD *)v11[1];
        v23 = 1;
        v18 = v12;
        if ( v13 == v12 )
        {
          v17 = 1;
        }
        else
        {
          do
          {
            v19 = v13;
            v25 = *v13;
            *v13 = 0;
            sub_16BCBB0((__int64 *)&v26, &v25, &v20);
            v14 = v23;
            a2 = &v24;
            v23 = 0;
            v24 = v14 | 1;
            sub_12BEC00(v27, (unsigned __int64 *)&v24, &v26);
            if ( (v23 & 1) != 0 || (v16 = v19, (v23 & 0xFFFFFFFFFFFFFFFELL) != 0) )
              sub_16BCAE0(&v23, (__int64)&v24, v15);
            v23 |= v27[0] | 1;
            if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v24, (__int64)&v24, v15);
            if ( (v26 & 1) != 0 || (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v26, (__int64)&v24, v15);
            if ( v25 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
              v16 = v19;
            }
            v13 = v16 + 1;
          }
          while ( v18 != v13 );
          v17 = v23 | 1;
        }
        v26 = v17;
        (*(void (__fastcall **)(_QWORD *))(*v11 + 8LL))(v11);
      }
      else
      {
        v27[0] = (unsigned __int64)v11;
        a2 = (__int64 *)v27;
        sub_16BCBB0((__int64 *)&v26, v27, &v20);
        if ( v27[0] )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v27[0] + 8LL))(v27[0]);
      }
      if ( (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v26 = v26 & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_16BCAE0(&v26, (__int64)a2, v8);
      }
      if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(&v22, (__int64)a2, v8);
    }
    result = v21;
    if ( (v21 & 1) != 0 || (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v21, (__int64)a2, v8);
  }
  else
  {
    *a1 = 0;
  }
  return result;
}
