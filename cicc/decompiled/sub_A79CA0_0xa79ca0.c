// Function: sub_A79CA0
// Address: 0xa79ca0
//
unsigned __int64 __fastcall sub_A79CA0(__int64 *a1, unsigned int *a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  unsigned int *v5; // r13
  __int64 *v6; // rbx
  unsigned int *v7; // r15
  unsigned int v8; // r14d
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 *v13; // rsi
  __int64 v14; // r10
  __int64 v15; // rax
  int v16; // edx
  int *v17; // rax
  int *v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // r8
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // [rsp-120h] [rbp-120h]
  unsigned __int64 v23; // [rsp-118h] [rbp-118h]
  __int64 *v24; // [rsp-100h] [rbp-100h]
  unsigned __int64 v25; // [rsp-100h] [rbp-100h]
  __int64 v26; // [rsp-100h] [rbp-100h]
  __int64 *v27; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v28; // [rsp-F0h] [rbp-F0h]
  __int64 v29; // [rsp-E8h] [rbp-E8h] BYREF
  int *v30; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v31; // [rsp-C0h] [rbp-C0h]
  _BYTE v32[184]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( !a3 )
    return 0;
  v4 = 4 * a3;
  v30 = (int *)v32;
  v31 = 0x800000000LL;
  if ( a2 == &a2[v4] )
  {
    v18 = (int *)v32;
    v19 = 0;
  }
  else
  {
    v5 = &a2[v4];
    v6 = &v29;
    v7 = a2;
    do
    {
      v8 = *v7;
      v9 = 4;
      v10 = 0;
      v28 = 0x400000000LL;
      v11 = v6;
      v27 = v6;
      while ( 1 )
      {
        v12 = *((_QWORD *)v7 + 1);
        if ( v10 + 1 > v9 )
        {
          v24 = v11;
          sub_C8D5F0(&v27, v11, v10 + 1, 8);
          v10 = (unsigned int)v28;
          v11 = v24;
        }
        v7 += 4;
        v27[v10] = v12;
        v10 = (unsigned int)(v28 + 1);
        LODWORD(v28) = v28 + 1;
        if ( v7 == v5 || v8 != *v7 )
          break;
        v9 = HIDWORD(v28);
      }
      v13 = v27;
      v6 = v11;
      v14 = sub_A79C90(a1, v27, v10);
      v15 = (unsigned int)v31;
      v16 = v31;
      if ( (unsigned int)v31 >= (unsigned __int64)HIDWORD(v31) )
      {
        v20 = v23 & 0xFFFFFFFF00000000LL | v8;
        v23 = v20;
        if ( HIDWORD(v31) < (unsigned __int64)(unsigned int)v31 + 1 )
        {
          v13 = (__int64 *)v32;
          v22 = v20;
          v26 = v14;
          sub_C8D5F0(&v30, v32, (unsigned int)v31 + 1LL, 16);
          v15 = (unsigned int)v31;
          v20 = v22;
          v14 = v26;
        }
        v21 = (unsigned __int64 *)&v30[4 * v15];
        *v21 = v20;
        v21[1] = v14;
        LODWORD(v31) = v31 + 1;
      }
      else
      {
        v17 = &v30[4 * (unsigned int)v31];
        if ( v17 )
        {
          *v17 = v8;
          *((_QWORD *)v17 + 1) = v14;
          v16 = v31;
        }
        LODWORD(v31) = v16 + 1;
      }
      if ( v27 != v6 )
        _libc_free(v27, v13);
    }
    while ( v5 != v7 );
    v18 = v30;
    v19 = (unsigned int)v31;
  }
  result = sub_A78010(a1, v18, v19);
  if ( v30 != (int *)v32 )
  {
    v25 = result;
    _libc_free(v30, v18);
    return v25;
  }
  return result;
}
