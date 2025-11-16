// Function: sub_31575F0
// Address: 0x31575f0
//
__int64 __fastcall sub_31575F0(__int64 a1, __int64 a2, const void *a3, size_t a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  char *v11; // r8
  const char *v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rcx
  size_t v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  size_t v18; // rdi
  const char *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r12
  __int64 v23; // [rsp+0h] [rbp-E0h]
  __int64 v24; // [rsp+8h] [rbp-D8h]
  char *v27; // [rsp+18h] [rbp-C8h]
  const char *v28[4]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v29; // [rsp+40h] [rbp-A0h]
  const char *v30; // [rsp+50h] [rbp-90h] BYREF
  size_t v31; // [rsp+58h] [rbp-88h]
  unsigned __int64 v32; // [rsp+60h] [rbp-80h]
  _BYTE v33[120]; // [rsp+68h] [rbp-78h] BYREF

  v30 = v33;
  v31 = 0;
  v32 = 60;
  v8 = sub_B2F730(a2);
  v9 = a5;
  switch ( *(_DWORD *)(v8 + 24) )
  {
    case 0:
      v15 = v31;
      goto LABEL_8;
    case 1:
    case 3:
      v10 = 2;
      v11 = ".L";
      goto LABEL_3;
    case 2:
    case 4:
      v10 = 1;
      v11 = "L";
      goto LABEL_3;
    case 5:
      v10 = 2;
      v11 = "L#";
      goto LABEL_3;
    case 6:
      v10 = 1;
      v11 = "$";
      goto LABEL_3;
    case 7:
      v10 = 3;
      v11 = "L..";
LABEL_3:
      if ( v10 + v31 > v32 )
      {
        v23 = a5;
        v24 = v10;
        v27 = v11;
        sub_C8D290((__int64)&v30, v33, v10 + v31, 1u, (__int64)v11, v9);
        v11 = v27;
        v10 = v24;
        v9 = v23;
      }
      v12 = &v30[v31];
      v13 = 0;
      do
      {
        v14 = v13++;
        v12[v14] = v11[v14];
      }
      while ( v13 < (unsigned int)v10 );
      v15 = v10 + v31;
LABEL_8:
      v31 = v15;
      sub_23CF320(v9, (__int64)&v30, a2, *(_QWORD *)(a1 + 928), 0);
      v18 = v31;
      if ( a4 + v31 > v32 )
      {
        sub_C8D290((__int64)&v30, v33, a4 + v31, 1u, v16, v17);
        v18 = v31;
      }
      v19 = v30;
      if ( a4 )
      {
        memcpy((void *)&v30[v18], a3, a4);
        v19 = v30;
        v18 = v31;
      }
      v20 = *(_QWORD *)(a1 + 920);
      v31 = a4 + v18;
      v28[1] = (const char *)(a4 + v18);
      v29 = 261;
      v28[0] = v19;
      v21 = sub_E6C460(v20, v28);
      if ( v30 != v33 )
        _libc_free((unsigned __int64)v30);
      return v21;
    default:
      BUG();
  }
}
