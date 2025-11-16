// Function: sub_37FAEF0
// Address: 0x37faef0
//
__int64 *__fastcall sub_37FAEF0(__int64 *a1, __int64 *a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r15
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 *v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 v16; // rax
  __int64 *v17; // [rsp+8h] [rbp-1E8h]
  __int64 *v18; // [rsp+20h] [rbp-1D0h]
  __int64 v19; // [rsp+38h] [rbp-1B8h] BYREF
  __int64 v20; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v21; // [rsp+48h] [rbp-1A8h] BYREF
  __int64 v22; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v23; // [rsp+58h] [rbp-198h] BYREF
  unsigned __int64 v24; // [rsp+60h] [rbp-190h] BYREF
  __int64 v25; // [rsp+68h] [rbp-188h] BYREF
  __int64 v26; // [rsp+70h] [rbp-180h] BYREF
  __int64 v27; // [rsp+78h] [rbp-178h] BYREF
  _QWORD v28[2]; // [rsp+80h] [rbp-170h] BYREF
  __int64 v29[2]; // [rsp+90h] [rbp-160h] BYREF
  int v30; // [rsp+A0h] [rbp-150h]
  _BYTE *v31; // [rsp+A8h] [rbp-148h]
  __int64 v32; // [rsp+B0h] [rbp-140h]
  __int64 v33; // [rsp+B8h] [rbp-138h]
  _BYTE v34[304]; // [rsp+C0h] [rbp-130h] BYREF

  v29[0] = (__int64)&off_4A3DA38;
  v4 = *a2;
  v29[1] = (__int64)a2;
  LODWORD(v28[0]) = 0;
  v30 = 0;
  v31 = v34;
  v32 = 0;
  v33 = 256;
  v5 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(v4 + 32))(a2, a3);
  v28[1] = v6;
  v28[0] = v5;
  sub_3707360(&v19, v28, a3, v29, 0);
  v7 = (_QWORD *)(v19 & 0xFFFFFFFFFFFFFFFELL);
  if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v19 = 0;
    v11 = (__int64 *)&unk_4F84052;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v7 + 48LL))(v7, &unk_4F84052) )
    {
      v12 = (__int64 *)v7[2];
      v13 = (__int64 *)v7[1];
      v23 = 1;
      v17 = v12;
      if ( v13 == v12 )
      {
        v16 = 1;
      }
      else
      {
        do
        {
          v18 = v13;
          v26 = *v13;
          *v13 = 0;
          sub_37F9BC0(&v25, &v26);
          v14 = v23;
          v11 = &v27;
          v23 = 0;
          v27 = v14 | 1;
          sub_9CDB40(&v24, (unsigned __int64 *)&v27, (unsigned __int64 *)&v25);
          if ( (v23 & 1) != 0 || (v15 = v18, (v23 & 0xFFFFFFFFFFFFFFFELL) != 0) )
            sub_C63C30(&v23, (__int64)&v27);
          v23 |= v24 | 1;
          if ( (v27 & 1) != 0 || (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v27, (__int64)&v27);
          if ( (v25 & 1) != 0 || (v25 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v25, (__int64)&v27);
          if ( v26 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
            v15 = v18;
          }
          v13 = v15 + 1;
        }
        while ( v17 != v13 );
        v16 = v23 | 1;
      }
      v26 = v16;
      (*(void (__fastcall **)(_QWORD *))(*v7 + 8LL))(v7);
    }
    else
    {
      v11 = &v27;
      v27 = (__int64)v7;
      sub_37F9BC0(&v26, &v27);
      if ( v27 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
    }
    if ( (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v22, (__int64)v11);
    if ( (v21 & 1) != 0 || (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v21, (__int64)v11);
    if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v20, (__int64)v11);
    *a1 = (__int64)(a1 + 2);
    sub_37F94B0(a1, "<unknown UDT>", (__int64)"");
    if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v19, (__int64)"<unknown UDT>");
  }
  else
  {
    v8 = v31;
    v9 = v32;
    *a1 = (__int64)(a1 + 2);
    sub_37F94B0(a1, v8, (__int64)&v8[v9]);
  }
  if ( v31 != v34 )
    _libc_free((unsigned __int64)v31);
  return a1;
}
