// Function: sub_23BD210
// Address: 0x23bd210
//
void __fastcall sub_23BD210(__int64 *a1, char a2, __int64 a3)
{
  __int64 v5; // rdi
  _QWORD *v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r13
  __int64 v14; // r14
  _QWORD *v15; // r8
  unsigned __int64 v16; // rdi
  __int64 v17; // r9
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r12
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 *v32; // rax
  __int64 v33; // rcx
  bool v34; // zf
  __int64 v35; // [rsp+8h] [rbp-C8h]
  _QWORD *v36; // [rsp+10h] [rbp-C0h]
  __int64 v37; // [rsp+18h] [rbp-B8h]
  int v38; // [rsp+2Ch] [rbp-A4h] BYREF
  unsigned __int64 **v39; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+38h] [rbp-98h]
  _QWORD v41[2]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 *v42; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 *v43; // [rsp+58h] [rbp-78h]
  __int64 v44; // [rsp+60h] [rbp-70h]
  unsigned __int64 v45; // [rsp+68h] [rbp-68h]
  __int64 v46; // [rsp+70h] [rbp-60h]
  __int64 v47; // [rsp+78h] [rbp-58h]
  _QWORD *v48; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v49[8]; // [rsp+90h] [rbp-40h] BYREF

  if ( a2 )
  {
    v38 = 0;
    LOBYTE(v41[0]) = 0;
    v39 = (unsigned __int64 **)v41;
    v40 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0x6000000000LL;
    v48 = v49;
    sub_23AEDD0((__int64 *)&v48, v41, (__int64)v41);
    v5 = *a1;
    v6 = (_QWORD *)a1[1];
    v39 = &v42;
    v40 = a3;
    v41[0] = &v38;
    sub_23BCBD0(v5, v6, (void (__fastcall *)(__int64, __int64, __int64))sub_23AECF0, (__int64)&v39);
    if ( v48 != v49 )
      j_j___libc_free_0((unsigned __int64)v48);
    v7 = v45;
    if ( HIDWORD(v46) && (_DWORD)v46 )
    {
      v8 = 0;
      v37 = 8LL * (unsigned int)v46;
      do
      {
        v9 = *(_QWORD *)(v7 + v8);
        if ( v9 && v9 != -8 )
        {
          v10 = *(_QWORD *)(v9 + 72);
          v11 = *(_QWORD *)v9 + 97LL;
          if ( *(_DWORD *)(v9 + 84) )
          {
            v12 = *(unsigned int *)(v9 + 80);
            if ( (_DWORD)v12 )
            {
              v13 = 8 * v12;
              v14 = 0;
              do
              {
                v15 = *(_QWORD **)(v10 + v14);
                if ( v15 != (_QWORD *)-8LL && v15 )
                {
                  v16 = v15[1];
                  v17 = *v15 + 41LL;
                  if ( (_QWORD *)v16 != v15 + 3 )
                  {
                    v35 = *v15 + 41LL;
                    v36 = v15;
                    j_j___libc_free_0(v16);
                    v17 = v35;
                    v15 = v36;
                  }
                  sub_C7D6A0((__int64)v15, v17, 8);
                  v10 = *(_QWORD *)(v9 + 72);
                }
                v14 += 8;
              }
              while ( v13 != v14 );
            }
          }
          _libc_free(v10);
          v18 = *(_QWORD *)(v9 + 40);
          if ( v18 != v9 + 56 )
            j_j___libc_free_0(v18);
          v19 = *(_QWORD *)(v9 + 8);
          if ( v19 != v9 + 24 )
            j_j___libc_free_0(v19);
          sub_C7D6A0(v9, v11, 8);
          v7 = v45;
        }
        v8 += 8;
      }
      while ( v37 != v8 );
    }
    _libc_free(v7);
    v20 = v43;
    v21 = v42;
    if ( v43 != v42 )
    {
      do
      {
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21);
        v21 += 4;
      }
      while ( v20 != v21 );
      v21 = v42;
    }
    if ( v21 )
      j_j___libc_free_0((unsigned __int64)v21);
  }
  else
  {
    v22 = a1[1];
    v23 = *(__int64 **)(v22 + 24);
    v24 = *(unsigned int *)(v22 + 32);
    v25 = *v23;
    if ( (_DWORD)v24 && (!v25 || v25 == -8) )
    {
      v26 = v23 + 1;
      do
      {
        do
          v25 = *v26++;
        while ( v25 == -8 );
      }
      while ( !v25 );
    }
    v27 = *a1;
    v28 = v25 + 8;
    v29 = *(__int64 **)(*a1 + 24);
    v30 = *(unsigned int *)(*a1 + 32);
    v31 = *v29;
    if ( (_DWORD)v30 && (v31 == -8 || !v31) )
    {
      v32 = v29 + 1;
      do
      {
        do
          v31 = *v32++;
        while ( v31 == -8 );
      }
      while ( !v31 );
    }
    v33 = v31 + 8;
    v34 = *(_QWORD *)(a3 + 16) == 0;
    LOBYTE(v39) = 0;
    LODWORD(v42) = 0;
    if ( v34 )
      sub_4263D6(v24, v30, v27);
    (*(void (__fastcall **)(__int64, unsigned __int64 ***, unsigned __int64 **, __int64, __int64))(a3 + 24))(
      a3,
      &v39,
      &v42,
      v33,
      v28);
  }
}
