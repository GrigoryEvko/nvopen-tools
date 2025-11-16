// Function: sub_256B620
// Address: 0x256b620
//
void __fastcall sub_256B620(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  __int64 v7; // r12
  _QWORD *i; // rax
  char *v9; // r15
  char *v10; // rax
  char *v11; // rbx
  __int64 *v12; // r12
  bool v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 *v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // [rsp+18h] [rbp-D8h]
  __int64 v21; // [rsp+20h] [rbp-D0h]
  __int64 v22; // [rsp+30h] [rbp-C0h]
  char v23; // [rsp+38h] [rbp-B8h]
  __int64 v24; // [rsp+38h] [rbp-B8h]
  char v25; // [rsp+4Fh] [rbp-A1h]
  __int64 v26; // [rsp+58h] [rbp-98h] BYREF
  char *v27; // [rsp+60h] [rbp-90h] BYREF
  __int64 v28; // [rsp+68h] [rbp-88h]
  _BYTE v29[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v30; // [rsp+90h] [rbp-60h] BYREF
  __int64 v31; // [rsp+98h] [rbp-58h] BYREF
  __int64 v32; // [rsp+A0h] [rbp-50h]
  __int64 *v33; // [rsp+A8h] [rbp-48h]
  __int64 *v34; // [rsp+B0h] [rbp-40h]
  __int64 v35; // [rsp+B8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 88) == 0;
  v27 = v29;
  v28 = 0x400000000LL;
  LODWORD(v31) = 0;
  v32 = 0;
  v33 = &v31;
  v34 = &v31;
  v35 = 0;
  v20 = a1 + 56;
  if ( v6 )
  {
    v7 = *(_QWORD *)a1;
    v25 = 1;
    v22 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  }
  else
  {
    v25 = 0;
    v7 = *(_QWORD *)(a1 + 72);
    v22 = a1 + 56;
  }
  if ( v25 )
    goto LABEL_14;
LABEL_4:
  if ( v22 != v7 )
  {
    for ( i = (_QWORD *)(v7 + 32); ; i = (_QWORD *)v7 )
    {
      a6 = *i + a2;
      v26 = a6;
      if ( v35 )
        goto LABEL_19;
      a3 = (unsigned __int64)v27;
      a4 = (unsigned int)v28;
      v9 = &v27[8 * (unsigned int)v28];
      if ( v27 == v9 )
      {
        if ( (unsigned int)v28 > 3uLL )
          goto LABEL_18;
      }
      else
      {
        v10 = v27;
        while ( a6 != *(_QWORD *)v10 )
        {
          v10 += 8;
          if ( v9 == v10 )
            goto LABEL_20;
        }
        if ( v10 != v9 )
          goto LABEL_12;
LABEL_20:
        if ( (unsigned int)v28 > 3uLL )
        {
          v21 = v7;
          v11 = &v27[8 * (unsigned int)v28];
          v12 = (__int64 *)v27;
          do
          {
            v15 = sub_FB52B0(&v30, &v31, v12);
            v17 = v16;
            if ( v16 )
            {
              v13 = v15 || v16 == &v31 || *v12 < v16[4];
              v23 = v13;
              v14 = sub_22077B0(0x28u);
              *(_QWORD *)(v14 + 32) = *v12;
              sub_220F040(v23, v14, v17, &v31);
              ++v35;
            }
            ++v12;
          }
          while ( v11 != (char *)v12 );
          v7 = v21;
LABEL_18:
          LODWORD(v28) = 0;
LABEL_19:
          sub_FADE70((__int64)&v30, &v26);
          goto LABEL_12;
        }
      }
      a3 = (unsigned int)v28 + 1LL;
      if ( a3 > HIDWORD(v28) )
      {
        v24 = a6;
        sub_C8D5F0((__int64)&v27, v29, a3, 8u, a5, a6);
        a3 = (unsigned int)v28;
        a6 = v24;
        v9 = &v27[8 * (unsigned int)v28];
      }
      *(_QWORD *)v9 = a6;
      LODWORD(v28) = v28 + 1;
LABEL_12:
      if ( !v25 )
      {
        v7 = sub_220EF30(v7);
        goto LABEL_4;
      }
      v7 += 8;
LABEL_14:
      if ( v22 == v7 )
        break;
    }
  }
  sub_25387F0(a1, &v27, a3, a4, a5, a6);
  sub_253B2D0(*(_QWORD *)(a1 + 64));
  v18 = v32;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 72) = v20;
  *(_QWORD *)(a1 + 80) = v20;
  if ( v18 )
  {
    v19 = v31;
    *(_QWORD *)(a1 + 64) = v18;
    *(_DWORD *)(a1 + 56) = v19;
    *(_QWORD *)(a1 + 72) = v33;
    *(_QWORD *)(a1 + 80) = v34;
    *(_QWORD *)(v18 + 8) = v20;
    *(_QWORD *)(a1 + 88) = v35;
  }
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
}
