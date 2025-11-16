// Function: sub_C64300
// Address: 0xc64300
//
__int64 __fastcall sub_C64300(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  int v4; // eax
  __int64 v5; // rdx
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // rax
  _QWORD *v13; // [rsp+8h] [rbp-D8h]
  _QWORD *v14; // [rsp+18h] [rbp-C8h]
  __int64 *v15; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v17; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v19; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-98h] BYREF
  __int64 v21; // [rsp+50h] [rbp-90h] BYREF
  __int64 v22; // [rsp+58h] [rbp-88h]
  __int64 v23[4]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v24[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v25; // [rsp+A0h] [rbp-40h]

  LODWORD(v21) = 0;
  v16 = 0;
  v22 = sub_2241E40();
  v15 = &v21;
  v2 = *a1;
  *a1 = 0;
  v3 = v2 & 0xFFFFFFFFFFFFFFFELL;
  if ( v3 )
  {
    v7 = (_QWORD *)v3;
    a2 = (__int64 *)&unk_4F84052;
    v17 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v3 + 48LL))(v3, &unk_4F84052) )
    {
      v8 = (_QWORD *)v7[2];
      v9 = (_QWORD *)v7[1];
      v18 = 1;
      v13 = v8;
      if ( v9 == v8 )
      {
        v12 = 1;
      }
      else
      {
        do
        {
          v14 = v9;
          v20 = *v9;
          *v9 = 0;
          sub_C63D10(v23, &v20, (__int64 *)&v15);
          v10 = v18;
          a2 = &v19;
          v18 = 0;
          v19 = v10 | 1;
          sub_9CDB40(v24, (unsigned __int64 *)&v19, (unsigned __int64 *)v23);
          if ( (v18 & 1) != 0 || (v11 = v14, (v18 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v18, (__int64)&v19);
          v18 |= v24[0] | 1;
          if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v19, (__int64)&v19);
          if ( (v23[0] & 1) != 0 || (v23[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(v23, (__int64)&v19);
          if ( v20 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
            v11 = v14;
          }
          v9 = v11 + 1;
        }
        while ( v13 != v9 );
        v12 = v18 | 1;
      }
      v23[0] = v12;
      (*(void (__fastcall **)(_QWORD *))(*v7 + 8LL))(v7);
    }
    else
    {
      v24[0] = (unsigned __int64)v7;
      a2 = (__int64 *)v24;
      sub_C63D10(v23, v24, (__int64 *)&v15);
      if ( v24[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v24[0] + 8LL))(v24[0]);
    }
    if ( (v23[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v17, (__int64)a2);
  }
  if ( (v16 & 1) != 0 || (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v16, (__int64)a2);
  v4 = sub_C63BB0();
  if ( v22 == v5 && (_DWORD)v21 == v4 )
  {
    (*(void (__fastcall **)(__int64 *))(*(_QWORD *)v22 + 32LL))(v23);
    v24[0] = (unsigned __int64)v23;
    v25 = 260;
    sub_C64D30(v24, 1);
  }
  return v21;
}
