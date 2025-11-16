// Function: sub_1219BC0
// Address: 0x1219bc0
//
__int64 __fastcall sub_1219BC0(__int64 a1, _QWORD **a2)
{
  __int64 v2; // r14
  unsigned int v3; // r12d
  unsigned __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // eax
  unsigned int v8; // ebx
  unsigned int v9; // eax
  const char *v10; // rdx
  unsigned int v11; // ebx
  const char *v12; // rax
  int v13; // eax
  unsigned int v14; // ebx
  const char *v15; // rax
  unsigned __int64 v16; // rsi
  unsigned int v17; // [rsp+Ch] [rbp-C4h]
  const char *v18; // [rsp+10h] [rbp-C0h]
  const void **v19; // [rsp+18h] [rbp-B8h]
  const char *v20; // [rsp+18h] [rbp-B8h]
  __int64 *v21; // [rsp+28h] [rbp-A8h] BYREF
  const char *v22; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-98h]
  const char *v24; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v25; // [rsp+48h] [rbp-88h]
  const char *v26; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+58h] [rbp-78h]
  const char *v28; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+68h] [rbp-68h]
  const char *v30; // [rsp+70h] [rbp-60h] BYREF
  __int64 v31; // [rsp+78h] [rbp-58h]
  __int64 v32; // [rsp+80h] [rbp-50h]
  unsigned int v33; // [rsp+88h] [rbp-48h]
  char v34; // [rsp+90h] [rbp-40h]
  char v35; // [rsp+91h] [rbp-3Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v23 = 1;
  v22 = 0;
  v25 = 1;
  v24 = 0;
  v21 = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '('") )
    goto LABEL_2;
  v35 = 1;
  v5 = *(_QWORD *)(a1 + 232);
  v30 = "expected type";
  v34 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v21, (int *)&v30, 0) )
    goto LABEL_2;
  if ( *((_BYTE *)v21 + 8) != 12 )
  {
    v35 = 1;
    v30 = "the range must have integer type!";
    v34 = 3;
    sub_11FD800(v2, v5, (__int64)&v30, 1);
LABEL_2:
    v3 = 1;
    goto LABEL_3;
  }
  v30 = (const char *)sub_BCAE30((__int64)v21);
  v31 = v6;
  v7 = sub_CA1930(&v30);
  v8 = v7;
  if ( *(_DWORD *)(a1 + 240) != 529 )
    goto LABEL_48;
  if ( v7 < *(_DWORD *)(a1 + 328) )
  {
LABEL_50:
    v35 = 1;
    v15 = "integer is too large for the bit width of specified type";
    goto LABEL_49;
  }
  v19 = (const void **)(a1 + 320);
  if ( *(_BYTE *)(a1 + 332) )
    sub_C449B0((__int64)&v30, (const void **)(a1 + 320), v7);
  else
    sub_C44830((__int64)&v30, v19, v7);
  v9 = v31;
  v10 = v30;
  if ( v23 > 0x40 && v22 )
  {
    v17 = v31;
    v18 = v30;
    j_j___libc_free_0_0(v22);
    v9 = v17;
    v10 = v18;
  }
  v22 = v10;
  v23 = v9;
  *(_DWORD *)(a1 + 240) = sub_1205200(v2);
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ','") )
    goto LABEL_2;
  if ( *(_DWORD *)(a1 + 240) != 529 )
  {
LABEL_48:
    v35 = 1;
    v15 = "expected integer";
LABEL_49:
    v30 = v15;
    v16 = *(_QWORD *)(a1 + 232);
    v34 = 3;
    v3 = 1;
    sub_11FD800(v2, v16, (__int64)&v30, 1);
    goto LABEL_3;
  }
  if ( v8 < *(_DWORD *)(a1 + 328) )
    goto LABEL_50;
  if ( *(_BYTE *)(a1 + 332) )
    sub_C449B0((__int64)&v30, v19, v8);
  else
    sub_C44830((__int64)&v30, v19, v8);
  v11 = v31;
  v12 = v30;
  if ( v25 > 0x40 && v24 )
  {
    v20 = v30;
    j_j___libc_free_0_0(v24);
    v12 = v20;
  }
  v25 = v11;
  v24 = v12;
  v13 = sub_1205200(v2);
  v14 = v23;
  *(_DWORD *)(a1 + 240) = v13;
  if ( v14 > 0x40 )
  {
    if ( !sub_C43C50((__int64)&v22, (const void **)&v24) || v14 == (unsigned int)sub_C444A0((__int64)&v22) )
      goto LABEL_31;
    goto LABEL_54;
  }
  if ( v22 == v24 && v22 )
  {
LABEL_54:
    v35 = 1;
    v15 = "the range represent the empty set but limits aren't 0!";
    goto LABEL_49;
  }
LABEL_31:
  v3 = sub_120AFE0(a1, 13, "expected ')'");
  if ( (_BYTE)v3 )
    goto LABEL_2;
  v29 = v25;
  if ( v25 > 0x40 )
    sub_C43780((__int64)&v28, (const void **)&v24);
  else
    v28 = v24;
  v27 = v23;
  if ( v23 > 0x40 )
    sub_C43780((__int64)&v26, (const void **)&v22);
  else
    v26 = v22;
  sub_AADC30((__int64)&v30, (__int64)&v26, (__int64 *)&v28);
  sub_A78C10(a2, (__int64)&v30);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( (unsigned int)v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
LABEL_3:
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return v3;
}
