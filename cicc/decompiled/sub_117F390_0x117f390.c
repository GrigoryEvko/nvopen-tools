// Function: sub_117F390
// Address: 0x117f390
//
__int64 __fastcall sub_117F390(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  _QWORD *v3; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdi
  unsigned int v12; // eax
  char v13; // al
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // rdi
  unsigned int v17; // eax
  _QWORD *v18; // rdi
  unsigned __int8 *v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned int v23; // eax
  _BYTE *v24; // rax
  const void **v25; // [rsp+8h] [rbp-88h] BYREF
  _QWORD *v26; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  unsigned int v31; // [rsp+38h] [rbp-58h]
  _QWORD *v32; // [rsp+40h] [rbp-50h] BYREF
  const void ***v33; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h]
  int v35; // [rsp+58h] [rbp-38h]

  v2 = 1;
  v3 = *(_QWORD **)a1;
  v25 = 0;
  if ( (_QWORD *)*v3 == a2 )
    return v2;
  v32 = a2;
  v33 = &v25;
  LOBYTE(v34) = 0;
  v5 = (_QWORD *)*v3;
  if ( *(_BYTE *)*v3 != 42 || a2 != (_QWORD *)*(v5 - 8) )
  {
LABEL_3:
    v6 = (_QWORD *)*v3;
    if ( *(_BYTE *)*v3 != 44 )
      goto LABEL_4;
    v19 = (unsigned __int8 *)*(v6 - 8);
    v20 = *v19;
    if ( (_BYTE)v20 == 17 )
    {
      v25 = (const void **)(v19 + 24);
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v19 + 1) + 8LL) - 17 > 1 || (unsigned __int8)v20 > 0x15u )
      {
LABEL_4:
        v32 = 0;
        v33 = (const void ***)a2;
        v7 = (_QWORD *)*v3;
        if ( *(_BYTE *)*v3 != 59 )
          return 0;
        v13 = sub_995B10(&v32, *(v7 - 8));
        v14 = *(v7 - 4);
        if ( (!v13 || (const void ***)v14 != v33)
          && (!(unsigned __int8)sub_995B10(&v32, v14) || (const void ***)*(v7 - 8) != v33) )
        {
          return 0;
        }
        sub_AB8340((__int64)&v32, *(_QWORD *)(a1 + 16));
        v15 = *(_QWORD *)(a1 + 16);
        if ( *(_DWORD *)(v15 + 8) > 0x40u && *(_QWORD *)v15 )
          j_j___libc_free_0_0(*(_QWORD *)v15);
        *(_QWORD *)v15 = v32;
        *(_DWORD *)(v15 + 8) = (_DWORD)v33;
        LODWORD(v33) = 0;
        if ( *(_DWORD *)(v15 + 24) <= 0x40u || (v16 = *(_QWORD *)(v15 + 16)) == 0 )
        {
          *(_QWORD *)(v15 + 16) = v34;
          *(_DWORD *)(v15 + 24) = v35;
          return 1;
        }
        j_j___libc_free_0_0(v16);
        v17 = (unsigned int)v33;
        *(_QWORD *)(v15 + 16) = v34;
        *(_DWORD *)(v15 + 24) = v35;
        if ( v17 <= 0x40 )
          return 1;
        v18 = v32;
        if ( !v32 )
          return 1;
LABEL_39:
        j_j___libc_free_0_0(v18);
        return 1;
      }
      v24 = sub_AD7630((__int64)v19, 0, v20);
      if ( !v24 || *v24 != 17 )
      {
LABEL_45:
        v3 = *(_QWORD **)a1;
        goto LABEL_4;
      }
      v25 = (const void **)(v24 + 24);
    }
    if ( a2 == (_QWORD *)*(v6 - 4) )
    {
      **(_BYTE **)(a1 + 8) = 1;
      v27 = *((_DWORD *)v25 + 2);
      if ( v27 > 0x40 )
        sub_C43780((__int64)&v26, v25);
      else
        v26 = *v25;
      sub_AADBC0((__int64)&v28, (__int64 *)&v26);
      sub_AB51C0((__int64)&v32, (__int64)&v28, *(_QWORD *)(a1 + 16));
      v21 = *(_QWORD *)(a1 + 16);
      if ( *(_DWORD *)(v21 + 8) > 0x40u && *(_QWORD *)v21 )
        j_j___libc_free_0_0(*(_QWORD *)v21);
      *(_QWORD *)v21 = v32;
      *(_DWORD *)(v21 + 8) = (_DWORD)v33;
      LODWORD(v33) = 0;
      if ( *(_DWORD *)(v21 + 24) > 0x40u && (v22 = *(_QWORD *)(v21 + 16)) != 0 )
      {
        j_j___libc_free_0_0(v22);
        v23 = (unsigned int)v33;
        *(_QWORD *)(v21 + 16) = v34;
        *(_DWORD *)(v21 + 24) = v35;
        if ( v23 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
      }
      else
      {
        *(_QWORD *)(v21 + 16) = v34;
        *(_DWORD *)(v21 + 24) = v35;
      }
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
      if ( v27 <= 0x40 )
        return 1;
      v18 = v26;
      if ( !v26 )
        return 1;
      goto LABEL_39;
    }
    goto LABEL_45;
  }
  v2 = sub_991580((__int64)&v33, *(v5 - 4));
  if ( !(_BYTE)v2 )
  {
    v3 = *(_QWORD **)a1;
    goto LABEL_3;
  }
  **(_BYTE **)(a1 + 8) = 1;
  v9 = *(_QWORD *)(a1 + 16);
  v27 = *((_DWORD *)v25 + 2);
  if ( v27 > 0x40 )
    sub_C43780((__int64)&v26, v25);
  else
    v26 = *v25;
  sub_AADBC0((__int64)&v28, (__int64 *)&v26);
  sub_AB4F10((__int64)&v32, v9, (__int64)&v28);
  v10 = *(_QWORD *)(a1 + 16);
  if ( *(_DWORD *)(v10 + 8) > 0x40u && *(_QWORD *)v10 )
    j_j___libc_free_0_0(*(_QWORD *)v10);
  *(_QWORD *)v10 = v32;
  *(_DWORD *)(v10 + 8) = (_DWORD)v33;
  LODWORD(v33) = 0;
  if ( *(_DWORD *)(v10 + 24) > 0x40u && (v11 = *(_QWORD *)(v10 + 16)) != 0 )
  {
    j_j___libc_free_0_0(v11);
    v12 = (unsigned int)v33;
    *(_QWORD *)(v10 + 16) = v34;
    *(_DWORD *)(v10 + 24) = v35;
    if ( v12 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
  }
  else
  {
    *(_QWORD *)(v10 + 16) = v34;
    *(_DWORD *)(v10 + 24) = v35;
  }
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return v2;
}
