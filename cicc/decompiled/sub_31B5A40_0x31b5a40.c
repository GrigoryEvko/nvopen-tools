// Function: sub_31B5A40
// Address: 0x31b5a40
//
__int64 __fastcall sub_31B5A40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // r15
  __int64 v12; // rdx
  unsigned __int64 *v13; // r13
  __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  int v16; // eax
  unsigned __int64 *v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // r13
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rbx
  unsigned __int64 v26; // r13
  char *v28; // r13
  __int64 v30; // [rsp+20h] [rbp-100h]
  __int64 v31; // [rsp+28h] [rbp-F8h]
  __int64 v33; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-D8h]
  __int64 v36; // [rsp+50h] [rbp-D0h]
  __int64 v37; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+68h] [rbp-B8h]
  _BYTE v39[176]; // [rsp+70h] [rbp-B0h] BYREF

  v37 = (__int64)v39;
  v38 = 0x1000000000LL;
  v30 = *(_QWORD *)(a2 + 24);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD *)(v3 + 80);
  v31 = v3 + 72;
  if ( v3 + 72 != v4 )
  {
    do
    {
      v5 = v4 - 24;
      if ( !v4 )
        v5 = 0;
      v6 = sub_3186770(v30, v5);
      v33 = *(_QWORD *)(a3 + 16);
      v7 = *(_QWORD *)(a2 + 24);
      v8 = sub_22077B0(0xC0u);
      v11 = v8;
      if ( v8 )
        sub_371BA10(v8, v7, v33);
      v12 = (unsigned int)v38;
      v13 = &v34;
      v34 = v11;
      v14 = v37;
      v15 = (unsigned int)v38 + 1LL;
      v16 = v38;
      if ( v15 > HIDWORD(v38) )
      {
        if ( v37 > (unsigned __int64)&v34 || (unsigned __int64)&v34 >= v37 + 8 * (unsigned __int64)(unsigned int)v38 )
        {
          sub_31B5960((__int64)&v37, v15, (unsigned int)v38, v37, v9, v10);
          v12 = (unsigned int)v38;
          v14 = v37;
          v16 = v38;
        }
        else
        {
          v28 = (char *)&v34 - v37;
          sub_31B5960((__int64)&v37, v15, (unsigned int)v38, v37, v9, v10);
          v14 = v37;
          v12 = (unsigned int)v38;
          v13 = (unsigned __int64 *)&v28[v37];
          v16 = v38;
        }
      }
      v17 = (unsigned __int64 *)(v14 + 8 * v12);
      if ( v17 )
      {
        *v17 = *v13;
        *v13 = 0;
        v11 = v34;
        v16 = v38;
      }
      v18 = (unsigned int)(v16 + 1);
      LODWORD(v38) = v18;
      if ( v11 )
      {
        sub_371BB90(v11);
        j_j___libc_free_0(v11);
        v18 = (unsigned int)v38;
      }
      v19 = (_QWORD *)(v37 + 8 * v18 - 8);
      sub_371B570(&v34, v6);
      v20 = *(_QWORD *)(v6 + 16) + 48LL;
      while ( v20 != v35 )
      {
        v21 = sub_371B3B0(&v34, v35, v36);
        sub_371C3E0(*v19, v21);
        sub_371B2F0(&v34);
      }
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v31 != v4 );
    v22 = (__int64 *)v37;
    v23 = (__int64 *)(v37 + 8LL * (unsigned int)v38);
    if ( (__int64 *)v37 != v23 )
    {
      do
      {
        v24 = *v22++;
        sub_318D2B0(a1 + 40, v24, a3);
      }
      while ( v23 != v22 );
      v25 = v37;
      v23 = (__int64 *)(v37 + 8LL * (unsigned int)v38);
      if ( (__int64 *)v37 != v23 )
      {
        do
        {
          v26 = *--v23;
          if ( v26 )
          {
            sub_371BB90(v26);
            j_j___libc_free_0(v26);
          }
        }
        while ( (__int64 *)v25 != v23 );
        v23 = (__int64 *)v37;
      }
    }
    if ( v23 != (__int64 *)v39 )
      _libc_free((unsigned __int64)v23);
  }
  return 0;
}
