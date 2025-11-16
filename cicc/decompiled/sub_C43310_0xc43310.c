// Function: sub_C43310
// Address: 0xc43310
//
__int64 __fastcall sub_C43310(void **a1, void *a2, unsigned __int64 a3, unsigned __int64 a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // r8
  unsigned __int64 v8; // r8
  _QWORD *v9; // r12
  __int64 *v10; // rax
  __int64 *v11; // rcx
  __int64 *v12; // rcx
  __int64 v13; // rax
  __int64 result; // rax
  __int64 *v15; // [rsp+8h] [rbp-98h]
  __int64 *v16; // [rsp+18h] [rbp-88h]
  __int64 v17; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h] BYREF
  __int64 v19; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v20; // [rsp+38h] [rbp-68h]
  unsigned __int64 v21; // [rsp+40h] [rbp-60h] BYREF
  __int64 v22; // [rsp+48h] [rbp-58h] BYREF
  __int64 v23; // [rsp+50h] [rbp-50h] BYREF
  __int64 v24; // [rsp+58h] [rbp-48h] BYREF
  __int64 v25; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int8 v26; // [rsp+68h] [rbp-38h]

  if ( a2 == sub_C33340() )
    sub_C3C460(a1, (__int64)a2);
  else
    sub_C37380(a1, (__int64)a2);
  sub_C43000((__int64)&v25, a1, a3, a4, 1u);
  v6 = v26;
  v26 &= ~2u;
  if ( (v6 & 1) != 0 )
  {
    v7 = v25;
    v17 = 0;
    v25 = 0;
    v8 = v7 & 0xFFFFFFFFFFFFFFFELL;
    v18 = 0;
    v9 = (_QWORD *)v8;
    if ( v8 )
    {
      v19 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v8 + 48LL))(v8, &unk_4F84052) )
      {
        v10 = (__int64 *)v9[2];
        v11 = (__int64 *)v9[1];
        v20 = 1;
        v15 = v10;
        if ( v11 == v10 )
        {
          v13 = 1;
        }
        else
        {
          do
          {
            v16 = v11;
            v23 = *v11;
            *v11 = 0;
            sub_C31FA0(&v22, &v23);
            v24 = v20 | 1;
            sub_9CDB40(&v21, (unsigned __int64 *)&v24, (unsigned __int64 *)&v22);
            v12 = v16;
            v20 = v21 | 1;
            if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v24);
            if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v22);
            if ( v23 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
              v12 = v16;
            }
            v11 = v12 + 1;
          }
          while ( v15 != v11 );
          v13 = v20 | 1;
        }
        v23 = v13;
        (*(void (__fastcall **)(_QWORD *))(*v9 + 8LL))(v9);
      }
      else
      {
        v24 = (__int64)v9;
        sub_C31FA0(&v23, &v24);
        if ( v24 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
      }
      if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v19);
    }
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  if ( (v18 & 1) != 0 || (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v18);
  if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v17);
  result = v26;
  if ( (v26 & 2) != 0 )
    sub_C432A0(&v25);
  if ( (v26 & 1) != 0 )
  {
    if ( v25 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
  }
  return result;
}
