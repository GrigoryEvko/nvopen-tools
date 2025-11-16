// Function: sub_2A8C840
// Address: 0x2a8c840
//
void __fastcall sub_2A8C840(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  __int64 i; // rbx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  char *v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v7 = (_QWORD *)a1[1];
  if ( v7 != (_QWORD *)a1[2] )
  {
    if ( v7 )
    {
      *v7 = v7 + 2;
      v7[1] = 0x100000000LL;
      if ( *(_DWORD *)(a2 + 8) )
        sub_2A8C4C0((__int64)v7, a2, a3, a4, a5, a6);
      v7 = (_QWORD *)a1[1];
    }
    a1[1] = (unsigned __int64)(v7 + 5);
    return;
  }
  v8 = (__int64)v7 - *a1;
  v29 = *a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 3);
  if ( v9 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v7 - *a1) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x3333333333333333LL * ((__int64)((__int64)v7 - *a1) >> 3);
  if ( v11 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v27 = 40;
      v26 = 0;
      v30 = 0;
      goto LABEL_13;
    }
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v23 = 40 * v12;
  }
  v25 = (char *)v7 - *a1;
  v24 = sub_22077B0(v23);
  v8 = (__int64)v25;
  v30 = v24;
  v26 = v24 + v23;
  v27 = v24 + 40;
LABEL_13:
  v13 = (_QWORD *)(v30 + v8);
  if ( v13 )
  {
    v10 = *(unsigned int *)(a2 + 8);
    *v13 = v13 + 2;
    v13[1] = 0x100000000LL;
    if ( (_DWORD)v10 )
      sub_2A8C4C0((__int64)v13, a2, (__int64)v13, v10, a5, a6);
  }
  v14 = v29;
  if ( v7 != (_QWORD *)v29 )
  {
    for ( i = v30; ; i = v16 )
    {
      if ( i
        && (*(_DWORD *)(i + 8) = 0,
            *(_QWORD *)i = i + 16,
            *(_DWORD *)(i + 12) = 1,
            v17 = *(unsigned int *)(v14 + 8),
            (_DWORD)v17) )
      {
        v28 = v14;
        sub_2A8BF20(i, v14, v17, v10, a5, a6);
        v16 = i + 40;
        v14 = v28 + 40;
        if ( v7 == (_QWORD *)(v28 + 40) )
        {
LABEL_23:
          v18 = i + 80;
          v19 = v29;
          v27 = v18;
          do
          {
            v20 = *(_QWORD *)v19;
            v21 = *(_QWORD *)v19 + 24LL * *(unsigned int *)(v19 + 8);
            if ( *(_QWORD *)v19 != v21 )
            {
              do
              {
                v21 -= 24LL;
                if ( *(_DWORD *)(v21 + 16) > 0x40u )
                {
                  v22 = *(_QWORD *)(v21 + 8);
                  if ( v22 )
                    j_j___libc_free_0_0(v22);
                }
              }
              while ( v20 != v21 );
              v21 = *(_QWORD *)v19;
            }
            if ( v21 != v19 + 16 )
              _libc_free(v21);
            v19 += 40LL;
          }
          while ( v7 != (_QWORD *)v19 );
          break;
        }
      }
      else
      {
        v14 += 40;
        v16 = i + 40;
        if ( v7 == (_QWORD *)v14 )
          goto LABEL_23;
      }
    }
  }
  if ( v29 )
    j_j___libc_free_0(v29);
  *a1 = v30;
  a1[1] = v27;
  a1[2] = v26;
}
