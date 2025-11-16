// Function: sub_12115D0
// Address: 0x12115d0
//
__int64 __fastcall sub_12115D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  unsigned __int64 v4; // rsi
  unsigned int v5; // edx
  char *v6; // rax
  unsigned __int64 v7; // rsi
  unsigned int v8; // edx
  char *v9; // rax
  unsigned int v10; // edx
  bool v11; // al
  int v12; // eax
  char v13; // r12
  bool v14; // cc
  unsigned int v15; // eax
  __int64 v16; // rdi
  char *v17; // rdx
  char *v18; // rdx
  unsigned int v19; // [rsp+8h] [rbp-B8h]
  char *v20; // [rsp+8h] [rbp-B8h]
  char *v21; // [rsp+10h] [rbp-B0h]
  __int64 v22; // [rsp+18h] [rbp-A8h]
  unsigned int v23; // [rsp+18h] [rbp-A8h]
  unsigned int v24; // [rsp+18h] [rbp-A8h]
  char *v25; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-98h]
  char v27; // [rsp+2Ch] [rbp-94h]
  char *v28; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-88h]
  char v30; // [rsp+3Ch] [rbp-84h]
  char *v31; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-78h]
  char *v33; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+58h] [rbp-68h]
  const char *v35; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+68h] [rbp-58h]
  __int64 v37; // [rsp+70h] [rbp-50h]
  int v38; // [rsp+78h] [rbp-48h]
  char v39; // [rsp+80h] [rbp-40h]
  char v40; // [rsp+81h] [rbp-3Fh]

  v27 = 0;
  v26 = 1;
  v25 = 0;
  v29 = 1;
  v28 = 0;
  v30 = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 459, "expected 'offset' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 6, "expected '[' here") )
  {
    goto LABEL_2;
  }
  if ( *(_DWORD *)(a1 + 240) != 529 )
  {
    v40 = 1;
    v4 = *(_QWORD *)(a1 + 232);
    v39 = 3;
    v35 = "expected integer";
    sub_11FD800(a1 + 176, v4, (__int64)&v35, 1);
LABEL_2:
    v2 = 1;
    goto LABEL_3;
  }
  v22 = a1 + 320;
  if ( *(_DWORD *)(a1 + 328) <= 0x40u )
  {
    v17 = *(char **)(a1 + 320);
    v26 = *(_DWORD *)(a1 + 328);
    v25 = v17;
  }
  else
  {
    sub_C43990((__int64)&v25, v22);
  }
  v27 = *(_BYTE *)(a1 + 332);
  if ( v27 )
    sub_C44AB0((__int64)&v35, (__int64)&v25, 0x40u);
  else
    sub_C44B10((__int64)&v35, &v25, 0x40u);
  v5 = v36;
  v6 = (char *)v35;
  if ( v26 > 0x40 && v25 )
  {
    v19 = v36;
    v21 = (char *)v35;
    j_j___libc_free_0_0(v25);
    v5 = v19;
    v6 = v21;
  }
  v26 = v5;
  v25 = v6;
  v27 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here") )
    goto LABEL_2;
  if ( *(_DWORD *)(a1 + 240) != 529 )
  {
    v40 = 1;
    v7 = *(_QWORD *)(a1 + 232);
    v39 = 3;
    v2 = 1;
    v35 = "expected integer";
    sub_11FD800(a1 + 176, v7, (__int64)&v35, 1);
    goto LABEL_3;
  }
  if ( v29 <= 0x40 && *(_DWORD *)(a1 + 328) <= 0x40u )
  {
    v18 = *(char **)(a1 + 320);
    v29 = *(_DWORD *)(a1 + 328);
    v28 = v18;
  }
  else
  {
    sub_C43990((__int64)&v28, v22);
  }
  v30 = *(_BYTE *)(a1 + 332);
  if ( v30 )
    sub_C44AB0((__int64)&v35, (__int64)&v28, 0x40u);
  else
    sub_C44B10((__int64)&v35, &v28, 0x40u);
  v8 = v36;
  v9 = (char *)v35;
  if ( v29 > 0x40 && v28 )
  {
    v20 = (char *)v35;
    v23 = v36;
    j_j___libc_free_0_0(v28);
    v9 = v20;
    v8 = v23;
  }
  v29 = v8;
  v28 = v9;
  v30 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v2 = sub_120AFE0(a1, 7, "expected ']' here");
  if ( (_BYTE)v2 )
    goto LABEL_2;
  sub_C46250((__int64)&v28);
  v10 = v26;
  if ( v26 <= 0x40 )
  {
    if ( v25 == v28 && v26 && v25 != (char *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26)) )
      goto LABEL_37;
LABEL_57:
    v34 = v29;
    if ( v29 > 0x40 )
    {
      sub_C43780((__int64)&v33, (const void **)&v28);
      v10 = v26;
    }
    else
    {
      v33 = v28;
    }
    v32 = v10;
    if ( v10 > 0x40 )
      sub_C43780((__int64)&v31, (const void **)&v25);
    else
      v31 = v25;
    v13 = 1;
    sub_AADC30((__int64)&v35, (__int64)&v31, (__int64 *)&v33);
    goto LABEL_38;
  }
  v24 = v26;
  v11 = sub_C43C50((__int64)&v25, (const void **)&v28);
  v10 = v24;
  if ( !v11 )
    goto LABEL_57;
  v12 = sub_C445E0((__int64)&v25);
  v10 = v24;
  if ( v24 == v12 )
    goto LABEL_57;
LABEL_37:
  v13 = 0;
  sub_AADB10((__int64)&v35, 0x40u, 0);
LABEL_38:
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  v14 = *(_DWORD *)(a2 + 24) <= 0x40u;
  *(_QWORD *)a2 = v35;
  v15 = v36;
  v36 = 0;
  *(_DWORD *)(a2 + 8) = v15;
  if ( v14 || (v16 = *(_QWORD *)(a2 + 16)) == 0 )
  {
    *(_QWORD *)(a2 + 16) = v37;
    *(_DWORD *)(a2 + 24) = v38;
  }
  else
  {
    j_j___libc_free_0_0(v16);
    v14 = v36 <= 0x40;
    *(_QWORD *)(a2 + 16) = v37;
    *(_DWORD *)(a2 + 24) = v38;
    if ( !v14 && v35 )
      j_j___libc_free_0_0(v35);
  }
  if ( v13 )
  {
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
  }
LABEL_3:
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return v2;
}
