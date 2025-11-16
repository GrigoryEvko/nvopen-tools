// Function: sub_22C8110
// Address: 0x22c8110
//
__int64 __fastcall sub_22C8110(__int64 a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  unsigned int v8; // eax
  unsigned int v9; // eax
  char v11; // r15
  unsigned int v12; // eax
  unsigned int v13; // esi
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 *v16; // rax
  unsigned int v20; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v21; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v23; // [rsp+40h] [rbp-B0h]
  unsigned int v24; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v25; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+58h] [rbp-98h]
  unsigned __int64 v27; // [rsp+60h] [rbp-90h]
  unsigned int v28; // [rsp+68h] [rbp-88h]
  unsigned __int64 v29; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v30; // [rsp+78h] [rbp-78h]
  unsigned __int64 v31; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+88h] [rbp-68h]
  unsigned __int64 v33; // [rsp+90h] [rbp-60h] BYREF
  signed __int64 v34; // [rsp+98h] [rbp-58h] BYREF
  unsigned __int64 v35; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v36; // [rsp+A8h] [rbp-48h] BYREF
  unsigned int v37; // [rsp+B0h] [rbp-40h]
  char v38; // [rsp+B8h] [rbp-38h]

  v8 = sub_BCB060(*(_QWORD *)(a4 + 8));
  sub_AADB10((__int64)&v21, v8, 1);
  if ( *(_BYTE *)a4 == 17 )
  {
    v30 = *(_DWORD *)(a4 + 32);
    if ( v30 > 0x40 )
      sub_C43780((__int64)&v29, (const void **)(a4 + 24));
    else
      v29 = *(_QWORD *)(a4 + 24);
    sub_AADBC0((__int64)&v33, (__int64 *)&v29);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    v21 = v33;
    v9 = v34;
    LODWORD(v34) = 0;
    v22 = v9;
    if ( v24 > 0x40 && v23 )
    {
      j_j___libc_free_0_0(v23);
      v23 = v35;
      v24 = v36;
      if ( (unsigned int)v34 > 0x40 && v33 )
      {
        j_j___libc_free_0_0(v33);
        if ( v30 <= 0x40 )
          goto LABEL_12;
LABEL_33:
        if ( v29 )
          j_j___libc_free_0_0(v29);
        goto LABEL_12;
      }
    }
    else
    {
      v23 = v35;
      v24 = v36;
    }
    if ( v30 > 0x40 )
      goto LABEL_33;
  }
  else if ( a7 )
  {
    sub_22C7100((__int64)&v33, a2, a4, *(_QWORD *)(a6 + 40), a6);
    if ( !v38 )
    {
      *(_BYTE *)(a1 + 40) = 0;
      goto LABEL_24;
    }
    v11 = v33;
    if ( (_BYTE)v33 == 4
      || (v12 = sub_BCB060(*(_QWORD *)(a4 + 8)), v13 = v12, v11 == 5)
      && (v20 = v12, v16 = sub_9876C0(&v34), v11 = v33, v13 = v20, v16) )
    {
      v30 = v35;
      if ( (unsigned int)v35 > 0x40 )
        sub_C43780((__int64)&v29, (const void **)&v34);
      else
        v29 = v34;
      v32 = v37;
      if ( v37 > 0x40 )
        sub_C43780((__int64)&v31, (const void **)&v36);
      else
        v31 = v36;
    }
    else if ( v11 == 2 )
    {
      sub_AD8380((__int64)&v29, v34);
    }
    else if ( v11 )
    {
      sub_AADB10((__int64)&v29, v13, 1);
    }
    else
    {
      sub_AADB10((__int64)&v29, v13, 0);
    }
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    v21 = v29;
    v14 = v30;
    v30 = 0;
    v22 = v14;
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    v23 = v31;
    v15 = v32;
    v32 = 0;
    v24 = v15;
    sub_969240((__int64 *)&v31);
    sub_969240((__int64 *)&v29);
    if ( v38 )
    {
      v38 = 0;
      sub_22C0090((unsigned __int8 *)&v33);
    }
  }
LABEL_12:
  sub_AB15A0((__int64)&v25, a3, (__int64)&v21);
  sub_AB1F90((__int64)&v29, (__int64 *)&v25, a5);
  sub_22C06B0((__int64)&v33, (__int64)&v29, 0);
  sub_22C0650(a1, (unsigned __int8 *)&v33);
  *(_BYTE *)(a1 + 40) = 1;
  sub_22C0090((unsigned __int8 *)&v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
LABEL_24:
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
