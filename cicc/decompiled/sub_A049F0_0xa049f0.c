// Function: sub_A049F0
// Address: 0xa049f0
//
__int64 __fastcall sub_A049F0(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rax
  __int64 result; // rax
  _QWORD *v4; // r12
  __int64 *v5; // rax
  __int64 *v6; // rcx
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // [rsp+0h] [rbp-80h]
  __int64 *v10; // [rsp+8h] [rbp-78h]
  __int64 v11; // [rsp+18h] [rbp-68h] BYREF
  __int64 v12; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v13; // [rsp+28h] [rbp-58h]
  unsigned __int64 v14; // [rsp+30h] [rbp-50h] BYREF
  __int64 v15; // [rsp+38h] [rbp-48h] BYREF
  __int64 v16; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v17[7]; // [rsp+48h] [rbp-38h] BYREF

  v1 = *a1;
  v11 = 0;
  *a1 = 0;
  v2 = v1 & 0xFFFFFFFFFFFFFFFELL;
  if ( v2 )
  {
    v4 = (_QWORD *)v2;
    v12 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v2 + 48LL))(v2, &unk_4F84052) )
    {
      v5 = (__int64 *)v4[2];
      v6 = (__int64 *)v4[1];
      v13 = 1;
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
          v16 = *v6;
          *v6 = 0;
          sub_A02750(&v15, &v16);
          v17[0] = v13 | 1;
          sub_9CDB40(&v14, v17, (unsigned __int64 *)&v15);
          v7 = v10;
          v13 = v14 | 1;
          if ( (v17[0] & 1) != 0 || (v17[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(v17);
          if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v15);
          if ( v16 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
            v7 = v10;
          }
          v6 = v7 + 1;
        }
        while ( v9 != v6 );
        v8 = v13 | 1;
      }
      v16 = v8;
      (*(void (__fastcall **)(_QWORD *))(*v4 + 8LL))(v4);
    }
    else
    {
      v17[0] = (unsigned __int64)v4;
      sub_A02750(&v16, v17);
      if ( v17[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v17[0] + 8LL))(v17[0]);
    }
    if ( (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v12 & 1) != 0 || (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v12);
  }
  result = v11;
  if ( (v11 & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v11);
  return result;
}
