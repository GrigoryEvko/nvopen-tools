// Function: sub_24FD1B0
// Address: 0x24fd1b0
//
_QWORD *__fastcall sub_24FD1B0(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  const char *v5; // rsi
  _BYTE *v6; // rax
  _BYTE *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int8 ***v11; // r8
  unsigned __int8 ***i; // r13
  unsigned __int8 **v13; // rbx
  unsigned __int8 *v14; // rax
  __int64 *v15; // rax
  unsigned __int8 *v16; // rax
  __int64 v17; // rbx
  unsigned __int8 *v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r12
  __int64 v21; // r10
  size_t v22; // rdx
  size_t v23; // rax
  size_t v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int8 *v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  unsigned __int8 ***v31; // [rsp+18h] [rbp-38h]

  v4 = *a3;
  if ( (sub_B6EA50(*a3)
     || (v26 = sub_B6F970(v4),
         (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v26 + 32LL))(
           v26,
           "annotation-remarks",
           18))
     || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v26 + 40LL))(
          v26,
          "annotation-remarks",
          18)
     || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v26 + 24LL))(
          v26,
          "annotation-remarks",
          18))
    && (v5 = "llvm.global.annotations", (v6 = sub_BA8CD0((__int64)a3, (__int64)"llvm.global.annotations", 0x17u, 0)) != 0)
    && (*((_DWORD *)v6 + 1) & 0x7FFFFFF) == 1 )
  {
    v8 = v6 - 32;
    if ( (v6[7] & 0x40) != 0 )
      v8 = (_BYTE *)*((_QWORD *)v6 - 1);
    v9 = *(_QWORD *)v8;
    v10 = 4LL * (*(_DWORD *)(*(_QWORD *)v8 + 4LL) & 0x7FFFFFF);
    if ( (*(_BYTE *)(*(_QWORD *)v8 + 7LL) & 0x40) != 0 )
    {
      v11 = *(unsigned __int8 ****)(v9 - 8);
      v31 = &v11[v10];
    }
    else
    {
      v31 = *(unsigned __int8 ****)v8;
      v11 = (unsigned __int8 ***)(v9 - v10 * 8);
    }
    for ( i = v11; v31 != i; i += 4 )
    {
      v13 = *i;
      if ( *(_BYTE *)*i == 10 && (*((_DWORD *)v13 + 1) & 0x7FFFFFF) == 4 )
      {
        v14 = sub_BD3990(*(v13 - 12), (__int64)v5);
        if ( *v14 <= 3u )
        {
          v15 = (v14[7] & 0x40) != 0
              ? (__int64 *)*((_QWORD *)v14 - 1)
              : (__int64 *)&v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
          v29 = *v15;
          if ( (unsigned int)*(unsigned __int8 *)*v15 - 15 <= 1 )
          {
            v16 = sub_BD3990(v13[-4 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)], (__int64)v5);
            if ( !*v16 )
            {
              v17 = *((_QWORD *)v16 + 10);
              v18 = v16 + 72;
              v19 = v29;
              if ( v16 + 72 == (unsigned __int8 *)v17 )
              {
                v20 = 0;
              }
              else
              {
                if ( !v17 )
                  BUG();
                while ( 1 )
                {
                  v20 = *(_QWORD *)(v17 + 32);
                  if ( v20 != v17 + 24 )
                    break;
                  v17 = *(_QWORD *)(v17 + 8);
                  if ( v18 == (unsigned __int8 *)v17 )
                    goto LABEL_24;
                  if ( !v17 )
                    BUG();
                }
              }
              while ( v18 != (unsigned __int8 *)v17 )
              {
                v21 = v20 - 24;
                v27 = v18;
                if ( !v20 )
                  v21 = 0;
                v30 = v19;
                v28 = v21;
                v5 = (const char *)sub_AC52D0(v19);
                v23 = v22;
                v24 = v22 - 1;
                if ( v23 < v24 )
                  v24 = 0;
                sub_B9D6F0(v28, (__int64)v5, v24);
                v20 = *(_QWORD *)(v20 + 8);
                v18 = v27;
                v19 = v30;
                while ( 1 )
                {
                  v25 = v17 - 24;
                  if ( !v17 )
                    v25 = 0;
                  if ( v20 != v25 + 48 )
                    break;
                  v17 = *(_QWORD *)(v17 + 8);
                  if ( v27 == (unsigned __int8 *)v17 )
                    goto LABEL_24;
                  if ( !v17 )
                    BUG();
                  v20 = *(_QWORD *)(v17 + 32);
                }
              }
            }
          }
        }
      }
LABEL_24:
      ;
    }
    memset(a1, 0, 0x60u);
    *((_BYTE *)a1 + 28) = 1;
    a1[1] = a1 + 4;
    *((_DWORD *)a1 + 4) = 2;
    a1[7] = a1 + 10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[6] = 0;
    a1[1] = a1 + 4;
    a1[7] = a1 + 10;
    a1[2] = 0x100000002LL;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
