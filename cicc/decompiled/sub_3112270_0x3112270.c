// Function: sub_3112270
// Address: 0x3112270
//
void __fastcall sub_3112270(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 *v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int *v9; // rbx
  int **v10; // rax
  int **v11; // rcx
  __int64 v12; // rax
  int **v13; // rcx
  __int64 v14; // rax
  int **v15; // [rsp+8h] [rbp-98h]
  int **v16; // [rsp+18h] [rbp-88h]
  _QWORD v17[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v18; // [rsp+30h] [rbp-70h] BYREF
  __int64 v19; // [rsp+38h] [rbp-68h] BYREF
  __int64 v20; // [rsp+40h] [rbp-60h] BYREF
  __int64 v21; // [rsp+48h] [rbp-58h] BYREF
  __int64 v22; // [rsp+50h] [rbp-50h] BYREF
  int *v23; // [rsp+58h] [rbp-48h] BYREF
  __int64 v24; // [rsp+60h] [rbp-40h] BYREF
  int *v25[7]; // [rsp+68h] [rbp-38h] BYREF

  v4 = *a1;
  v17[0] = a2;
  v17[1] = a3;
  v5 = v4 & 0xFFFFFFFFFFFFFFFELL;
  if ( v5 )
  {
    v6 = (__int64 *)&unk_5031F50;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v5 + 48LL))(v5, &unk_5031F50) )
    {
      v19 = 0;
      v18 = v17;
      v7 = *a1;
      *a1 = 0;
      v8 = v7 & 0xFFFFFFFFFFFFFFFELL;
      v9 = (int *)v8;
      if ( v8 )
      {
        v20 = 0;
        v6 = (__int64 *)&unk_4F84052;
        if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v8 + 48LL))(v8, &unk_4F84052) )
        {
          v10 = (int **)*((_QWORD *)v9 + 2);
          v11 = (int **)*((_QWORD *)v9 + 1);
          v21 = 1;
          v15 = v10;
          if ( v11 == v10 )
          {
            v14 = 1;
          }
          else
          {
            do
            {
              v16 = v11;
              v23 = *v11;
              *v11 = 0;
              sub_3112020(&v24, &v23, (__int64 *)&v18);
              v12 = v21;
              v6 = &v22;
              v21 = 0;
              v22 = v12 | 1;
              sub_9CDB40((unsigned __int64 *)v25, (unsigned __int64 *)&v22, (unsigned __int64 *)&v24);
              if ( (v21 & 1) != 0 || (v13 = v16, (v21 & 0xFFFFFFFFFFFFFFFELL) != 0) )
                sub_C63C30(&v21, (__int64)&v22);
              v21 |= (unsigned __int64)v25[0] | 1;
              if ( (v22 & 1) != 0 || (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v22, (__int64)&v22);
              if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v24, (__int64)&v22);
              if ( v23 )
              {
                (*(void (__fastcall **)(int *))(*(_QWORD *)v23 + 8LL))(v23);
                v13 = v16;
              }
              v11 = v13 + 1;
            }
            while ( v15 != v11 );
            v14 = v21 | 1;
          }
          v24 = v14;
          (*(void (__fastcall **)(int *))(*(_QWORD *)v9 + 8LL))(v9);
        }
        else
        {
          v25[0] = v9;
          v6 = (__int64 *)v25;
          sub_3112020(&v24, v25, (__int64 *)&v18);
          if ( v25[0] )
            (*(void (__fastcall **)(int *))(*(_QWORD *)v25[0] + 8LL))(v25[0]);
        }
        if ( (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          BUG();
        if ( (v20 & 1) != 0 || (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v20, (__int64)v6);
      }
      if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v19, (__int64)v6);
    }
  }
}
