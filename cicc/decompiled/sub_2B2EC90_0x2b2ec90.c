// Function: sub_2B2EC90
// Address: 0x2b2ec90
//
__int64 __fastcall sub_2B2EC90(__int64 ***a1, __int64 a2, char *a3, __int64 a4)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  char *v6; // r15
  __int64 v7; // rcx
  char *v9; // rbx
  char *v10; // r12
  _QWORD *v11; // rdi
  _QWORD *v12; // r14
  _QWORD *v13; // rdi
  _QWORD *v14; // r14
  _QWORD *v15; // rdi
  _QWORD *v16; // r14
  _QWORD *v17; // r14
  int v18; // esi
  __int64 **v19; // rdi
  int v20; // eax
  __int64 v21; // rax
  __int64 **v22; // r14
  __int64 v23; // r15
  __int64 **v24; // rax
  __int64 **v25; // rax
  __int64 v26; // r13
  __int64 v27; // rax
  char v28; // bl
  _QWORD *v29; // rax
  __int64 v30; // r12
  __int64 *v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rdx
  unsigned int v34; // esi
  _QWORD *v36; // r12
  _QWORD *v37; // r12
  _QWORD *v38; // r12
  char v40[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v41; // [rsp+40h] [rbp-70h]
  __int64 v42; // [rsp+50h] [rbp-60h] BYREF
  __int64 v43; // [rsp+58h] [rbp-58h]
  __int16 v44; // [rsp+70h] [rbp-40h]

  v4 = a4 << 6;
  v5 = v4 >> 6;
  v6 = &a3[v4];
  v7 = v4 >> 8;
  v9 = a3;
  if ( v7 > 0 )
  {
    v10 = &a3[256 * v7];
    while ( 1 )
    {
      v17 = (_QWORD *)(*(_QWORD *)v9 + 8LL * *((unsigned int *)v9 + 2));
      if ( v17 != sub_2B14D60(*(_QWORD **)v9, (__int64)v17) )
        goto LABEL_8;
      v11 = (_QWORD *)*((_QWORD *)v9 + 8);
      v12 = &v11[*((unsigned int *)v9 + 18)];
      if ( v12 != sub_2B14D60(v11, (__int64)v12) )
      {
        v9 += 64;
        goto LABEL_8;
      }
      v13 = (_QWORD *)*((_QWORD *)v9 + 16);
      v14 = &v13[*((unsigned int *)v9 + 34)];
      if ( v14 != sub_2B14D60(v13, (__int64)v14) )
      {
        v9 += 128;
        goto LABEL_8;
      }
      v15 = (_QWORD *)*((_QWORD *)v9 + 24);
      v16 = &v15[*((unsigned int *)v9 + 50)];
      if ( v16 != sub_2B14D60(v15, (__int64)v16) )
      {
        v9 += 192;
        goto LABEL_8;
      }
      v9 += 256;
      if ( v10 == v9 )
      {
        v5 = (v6 - v9) >> 6;
        break;
      }
    }
  }
  if ( v5 == 2 )
    goto LABEL_29;
  if ( v5 == 3 )
  {
    v36 = (_QWORD *)(*(_QWORD *)v9 + 8LL * *((unsigned int *)v9 + 2));
    if ( v36 != sub_2B14D60(*(_QWORD **)v9, (__int64)v36) )
      goto LABEL_8;
    v9 += 64;
LABEL_29:
    v37 = (_QWORD *)(*(_QWORD *)v9 + 8LL * *((unsigned int *)v9 + 2));
    if ( v37 != sub_2B14D60(*(_QWORD **)v9, (__int64)v37) )
      goto LABEL_8;
    v9 += 64;
    goto LABEL_31;
  }
  if ( v5 != 1 )
    goto LABEL_22;
LABEL_31:
  v38 = (_QWORD *)(*(_QWORD *)v9 + 8LL * *((unsigned int *)v9 + 2));
  if ( v38 == sub_2B14D60(*(_QWORD **)v9, (__int64)v38) )
    goto LABEL_22;
LABEL_8:
  if ( v6 == v9 )
  {
LABEL_22:
    LOBYTE(v43) = 0;
    return v42;
  }
  v18 = *(_DWORD *)(a2 + 120);
  if ( !v18 )
  {
    v19 = *a1;
    v18 = *(_DWORD *)(a2 + 8);
    v20 = *((unsigned __int8 *)*a1 + 8);
    if ( (_BYTE)v20 == 17 )
      goto LABEL_11;
LABEL_25:
    if ( (unsigned int)(v20 - 17) > 1 )
      goto LABEL_13;
    goto LABEL_12;
  }
  v19 = *a1;
  v20 = *((unsigned __int8 *)*a1 + 8);
  if ( (_BYTE)v20 != 17 )
    goto LABEL_25;
LABEL_11:
  v18 *= *((_DWORD *)v19 + 8);
LABEL_12:
  v19 = (__int64 **)*v19[2];
LABEL_13:
  v21 = sub_BCDA70((__int64 *)v19, v18);
  v22 = a1[14];
  v23 = v21;
  v24 = *a1;
  v41 = 257;
  v25 = (__int64 **)sub_BCE3C0(*v24, 0);
  v26 = sub_ACADE0(v25);
  v27 = sub_AA4E30((__int64)v22[6]);
  v28 = sub_AE5020(v27, v23);
  v44 = 257;
  v29 = sub_BD2C40(80, 1u);
  v30 = (__int64)v29;
  if ( v29 )
    sub_B4D190((__int64)v29, v23, v26, (__int64)&v42, 0, v28, 0, 0);
  (*(void (__fastcall **)(__int64 *, __int64, char *, __int64 *, __int64 *))(*v22[11] + 16))(
    v22[11],
    v30,
    v40,
    v22[7],
    v22[8]);
  v31 = *v22;
  v32 = (__int64)&(*v22)[2 * *((unsigned int *)v22 + 2)];
  if ( *v22 != (__int64 *)v32 )
  {
    do
    {
      v33 = v31[1];
      v34 = *(_DWORD *)v31;
      v31 += 2;
      sub_B99FD0(v30, v34, v33);
    }
    while ( (__int64 *)v32 != v31 );
  }
  v42 = v30;
  LOBYTE(v43) = 1;
  return v42;
}
