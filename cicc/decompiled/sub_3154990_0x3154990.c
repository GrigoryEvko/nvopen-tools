// Function: sub_3154990
// Address: 0x3154990
//
__int64 __fastcall sub_3154990(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rsi
  unsigned int v6; // r12d
  bool v8; // zf
  unsigned __int64 v9; // rax
  _QWORD *v10; // rbx
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 *v13; // rcx
  __int64 v14; // rax
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
  char v26; // [rsp+68h] [rbp-38h]

  v5 = (__int64 *)a1;
  sub_3154960((__int64)&v25, a1, a3, a4);
  v6 = v26 & 1;
  if ( (v26 & 1) != 0 )
  {
    v17 = 0;
    v26 &= ~2u;
    v18 = 0;
    v9 = v25 & 0xFFFFFFFFFFFFFFFELL;
    v8 = (v25 & 0xFFFFFFFFFFFFFFFELL) == 0;
    v25 = 0;
    v10 = (_QWORD *)v9;
    if ( !v8 )
    {
      v19 = 0;
      v5 = (__int64 *)&unk_4F84052;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v9 + 48LL))(v9, &unk_4F84052) )
      {
        v11 = (__int64 *)v10[2];
        v12 = (__int64 *)v10[1];
        v20 = 1;
        v15 = v11;
        if ( v12 == v11 )
        {
          v14 = 1;
        }
        else
        {
          do
          {
            v16 = v12;
            v23 = *v12;
            *v12 = 0;
            sub_3154270(&v22, &v23);
            v5 = &v24;
            v24 = v20 | 1;
            sub_9CDB40(&v21, (unsigned __int64 *)&v24, (unsigned __int64 *)&v22);
            v13 = v16;
            v20 = v21 | 1;
            if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v24, (__int64)&v24);
            if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v22, (__int64)&v24);
            if ( v23 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
              v13 = v16;
            }
            v12 = v13 + 1;
          }
          while ( v15 != v12 );
          v14 = v20 | 1;
        }
        v23 = v14;
        (*(void (__fastcall **)(_QWORD *))(*v10 + 8LL))(v10);
      }
      else
      {
        v5 = &v24;
        v24 = (__int64)v10;
        sub_3154270(&v23, &v24);
        if ( v24 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
      }
      if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v19, (__int64)v5);
    }
    if ( (v18 & 1) != 0 || (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v18, (__int64)v5);
    if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v17, (__int64)v5);
    v6 = (v26 & 2) != 0;
    if ( (v26 & 2) != 0 )
      sub_9CEF10(&v25);
    if ( (v26 & 1) != 0 && v25 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
  }
  else if ( (_DWORD)v25 == 2 && (unsigned int)(HIDWORD(v25) - 8) <= 5 )
  {
    *a2 = HIDWORD(v25);
    return 1;
  }
  return v6;
}
