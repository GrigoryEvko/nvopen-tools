// Function: sub_311FC60
// Address: 0x311fc60
//
void __fastcall sub_311FC60(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // r13
  char *v4; // r12
  __int64 v5; // rbx
  char *v6; // rbx
  char *v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  char *v19; // r14
  char *v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  char *v27; // [rsp+18h] [rbp-88h]
  __int64 v28; // [rsp+28h] [rbp-78h]
  char v29; // [rsp+3Fh] [rbp-61h] BYREF
  __int64 v30; // [rsp+40h] [rbp-60h] BYREF
  __int64 v31; // [rsp+48h] [rbp-58h] BYREF
  char *v32; // [rsp+50h] [rbp-50h] BYREF
  char *v33; // [rsp+58h] [rbp-48h]
  __int64 v34; // [rsp+60h] [rbp-40h]

  v2 = a2;
  v3 = a1;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  if ( (unsigned __int8)sub_CB4D10(a2, a2) )
  {
    v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 24LL))(a2);
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2) )
      v11 = 954437177 * ((v33 - v32) >> 4);
    if ( v11 )
    {
      v12 = (unsigned int)(v11 - 1);
      v13 = 0;
      v14 = a2;
      v15 = 1;
      v28 = v12 + 2;
      do
      {
        while ( 1 )
        {
          a2 = (unsigned int)(v15 - 1);
          v16 = v13 + 144;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 *))(*(_QWORD *)v14 + 32LL))(v14, a2, &v30) )
            break;
          v13 += 144;
          if ( ++v15 == v28 )
            goto LABEL_34;
        }
        v17 = (__int64)v32;
        v18 = 0x8E38E38E38E38E39LL * ((v33 - v32) >> 4);
        if ( v18 <= v15 - 1 )
        {
          if ( v15 > v18 )
          {
            sub_311F8E0((__int64 *)&v32, v15 - v18);
            v17 = (__int64)v32;
          }
          else if ( v15 < v18 )
          {
            v27 = &v32[v16];
            if ( v33 != &v32[v16] )
            {
              v24 = v13;
              v19 = v33;
              v20 = &v32[v16];
              do
              {
                v21 = *((_QWORD *)v20 + 10);
                if ( (char *)v21 != v20 + 96 )
                  _libc_free(v21);
                v22 = *((_QWORD *)v20 + 5);
                if ( (char *)v22 != v20 + 56 )
                  j_j___libc_free_0(v22);
                v23 = *((_QWORD *)v20 + 1);
                if ( (char *)v23 != v20 + 24 )
                  j_j___libc_free_0(v23);
                v20 += 144;
              }
              while ( v19 != v20 );
              v13 = v24;
              v17 = (__int64)v32;
              v33 = v27;
            }
          }
        }
        v26 = v17 + v13;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 104LL))(v14);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v14 + 120LL))(
               v14,
               "Hash",
               1,
               0,
               &v29,
               &v31) )
        {
          sub_311D010(v14, (_QWORD *)v26);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 128LL))(v14, v31);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v14 + 120LL))(
               v14,
               "FunctionName",
               1,
               0,
               &v29,
               &v31) )
        {
          sub_311EEB0(v14, (_QWORD *)(v26 + 8));
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 128LL))(v14, v31);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v14 + 120LL))(
               v14,
               "ModuleName",
               1,
               0,
               &v29,
               &v31) )
        {
          sub_311EEB0(v14, (_QWORD *)(v26 + 40));
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 128LL))(v14, v31);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v14 + 120LL))(
               v14,
               "InstCount",
               1,
               0,
               &v29,
               &v31) )
        {
          sub_311CE30(v14, (unsigned int *)(v26 + 72));
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 128LL))(v14, v31);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v14 + 120LL))(
               v14,
               "IndexOperandHashes",
               1,
               0,
               &v29,
               &v31) )
        {
          sub_311D330(v14, v26 + 80);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 128LL))(v14, v31);
        }
        v13 = v16;
        ++v15;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 112LL))(v14);
        a2 = v30;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 40LL))(v14, v30);
      }
      while ( v15 != v28 );
LABEL_34:
      v3 = a1;
      v2 = v14;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
  }
  v4 = v33;
  v5 = (__int64)v32;
  if ( v32 != v33 )
  {
    do
    {
      a2 = v5;
      v5 += 144;
      sub_311B700(*v3, (__int64 *)a2);
    }
    while ( v4 != (char *)v5 );
  }
  sub_CB0D90(v2, a2);
  v6 = v33;
  v7 = v32;
  if ( v33 != v32 )
  {
    do
    {
      v8 = *((_QWORD *)v7 + 10);
      if ( (char *)v8 != v7 + 96 )
        _libc_free(v8);
      v9 = *((_QWORD *)v7 + 5);
      if ( (char *)v9 != v7 + 56 )
        j_j___libc_free_0(v9);
      v10 = *((_QWORD *)v7 + 1);
      if ( (char *)v10 != v7 + 24 )
        j_j___libc_free_0(v10);
      v7 += 144;
    }
    while ( v6 != v7 );
    v7 = v32;
  }
  if ( v7 )
    j_j___libc_free_0((unsigned __int64)v7);
}
