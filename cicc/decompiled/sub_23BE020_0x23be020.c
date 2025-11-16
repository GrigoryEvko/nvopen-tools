// Function: sub_23BE020
// Address: 0x23be020
//
void __fastcall sub_23BE020(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  _QWORD *v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  __int64 v9; // rbx
  _QWORD *v10; // r12
  unsigned __int64 v11; // rdi
  __int64 v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r12
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rcx
  bool v25; // zf
  int v26; // [rsp+Ch] [rbp-A4h] BYREF
  unsigned __int64 **v27; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-98h]
  _QWORD v29[2]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 *v30; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 *v31; // [rsp+38h] [rbp-78h]
  __int64 v32; // [rsp+40h] [rbp-70h]
  unsigned __int64 v33; // [rsp+48h] [rbp-68h]
  __int64 v34; // [rsp+50h] [rbp-60h]
  __int64 v35; // [rsp+58h] [rbp-58h]
  _QWORD *v36; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v37[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( (_BYTE)a2 )
  {
    v26 = 0;
    v27 = (unsigned __int64 **)v29;
    sub_23AE760((__int64 *)&v27, byte_3F871B3, (__int64)byte_3F871B3);
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0x5000000000LL;
    v36 = v37;
    sub_23AEDD0((__int64 *)&v36, v27, (__int64)v27 + v28);
    if ( v27 != v29 )
      j_j___libc_free_0((unsigned __int64)v27);
    v5 = *a1;
    v6 = (_QWORD *)a1[1];
    v27 = &v30;
    v28 = a3;
    v29[0] = &v26;
    sub_23BD9E0(v5, v6, (void (__fastcall *)(__int64, __int64, __int64))sub_23AEC90, (__int64)&v27);
    if ( v36 != v37 )
      j_j___libc_free_0((unsigned __int64)v36);
    v7 = v33;
    if ( HIDWORD(v34) && (_DWORD)v34 )
    {
      v8 = 8LL * (unsigned int)v34;
      v9 = 0;
      do
      {
        v10 = *(_QWORD **)(v7 + v9);
        if ( v10 != (_QWORD *)-8LL && v10 )
        {
          v11 = v10[5];
          v12 = *v10 + 81LL;
          if ( (_QWORD *)v11 != v10 + 7 )
            j_j___libc_free_0(v11);
          v13 = v10[1];
          if ( (_QWORD *)v13 != v10 + 3 )
            j_j___libc_free_0(v13);
          sub_C7D6A0((__int64)v10, v12, 8);
          v7 = v33;
        }
        v9 += 8;
      }
      while ( v8 != v9 );
    }
    _libc_free(v7);
    v14 = v31;
    v15 = v30;
    if ( v31 != v30 )
    {
      do
      {
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
          j_j___libc_free_0(*v15);
        v15 += 4;
      }
      while ( v14 != v15 );
      v15 = v30;
    }
    if ( v15 )
      j_j___libc_free_0((unsigned __int64)v15);
  }
  else
  {
    v16 = a1[1];
    v17 = *(__int64 **)(v16 + 24);
    v18 = *v17;
    if ( *(_DWORD *)(v16 + 32) && (!v18 || v18 == -8) )
    {
      do
      {
        do
        {
          v18 = v17[1];
          ++v17;
        }
        while ( v18 == -8 );
      }
      while ( !v18 );
    }
    v19 = v18 + 8;
    v20 = *(__int64 **)(*a1 + 24);
    v21 = *(unsigned int *)(*a1 + 32);
    v22 = *v20;
    if ( (_DWORD)v21 && (!v22 || v22 == -8) )
    {
      v23 = v20 + 1;
      do
      {
        do
          v22 = *v23++;
        while ( v22 == -8 );
      }
      while ( !v22 );
    }
    v24 = v22 + 8;
    v25 = *(_QWORD *)(a3 + 16) == 0;
    LOBYTE(v27) = 0;
    LODWORD(v30) = 0;
    if ( v25 )
      sub_4263D6(a1, a2, v21);
    (*(void (__fastcall **)(__int64, unsigned __int64 ***, unsigned __int64 **, __int64, __int64))(a3 + 24))(
      a3,
      &v27,
      &v30,
      v24,
      v19);
  }
}
