// Function: sub_AAF050
// Address: 0xaaf050
//
__int64 __fastcall sub_AAF050(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r15
  unsigned int v6; // r13d
  unsigned int v7; // esi
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  int v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-78h]
  unsigned __int64 v30; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-68h]
  unsigned __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-58h]
  unsigned __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-48h]
  unsigned __int64 v36; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v37; // [rsp+58h] [rbp-38h]

  v3 = a2 + 16;
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 <= 0x40 )
  {
    v12 = *(_QWORD *)a2;
    if ( (*(_QWORD *)(a2 + 16) & *(_QWORD *)a2) == 0 )
    {
      if ( v12 )
        goto LABEL_10;
      goto LABEL_29;
    }
LABEL_25:
    sub_AADB10(a1, v6, 0);
    return a1;
  }
  if ( (unsigned __int8)sub_C446A0(a2, a2 + 16) )
    goto LABEL_25;
  if ( v6 != (unsigned int)sub_C444A0(a2) )
  {
    if ( !a3 )
    {
LABEL_35:
      v37 = v6;
      goto LABEL_36;
    }
    v7 = *(_DWORD *)(a2 + 24);
    v8 = *(_QWORD *)(a2 + 16);
    v9 = v7 - 1;
    v10 = 1LL << ((unsigned __int8)v7 - 1);
    if ( v7 <= 0x40 )
      goto LABEL_7;
    goto LABEL_6;
  }
LABEL_29:
  v16 = *(_DWORD *)(a2 + 24);
  if ( v16 > 0x40 )
  {
    v27 = *(_DWORD *)(a2 + 24);
    if ( v27 != (unsigned int)sub_C444A0(v3) )
    {
      if ( !a3 )
        goto LABEL_38;
      v9 = v27 - 1;
      v10 = 1LL << ((unsigned __int8)v27 - 1);
LABEL_6:
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (v9 >> 6));
      goto LABEL_7;
    }
LABEL_33:
    sub_AADB10(a1, v6, 1);
    return a1;
  }
  v8 = *(_QWORD *)(a2 + 16);
  if ( !v8 )
    goto LABEL_33;
  if ( !a3 )
    goto LABEL_38;
  v10 = 1LL << ((unsigned __int8)v16 - 1);
LABEL_7:
  if ( (v8 & v10) == 0 )
  {
    v11 = 1LL << ((unsigned __int8)v6 - 1);
    v12 = *(_QWORD *)a2;
    if ( v6 <= 0x40 )
    {
      if ( (v11 & v12) != 0 )
      {
LABEL_10:
        v37 = v6;
LABEL_11:
        v36 = v12;
        goto LABEL_12;
      }
LABEL_45:
      sub_9865C0((__int64)&v28, v3);
      sub_9865C0((__int64)&v36, a2);
      if ( v37 > 0x40 )
      {
        sub_C43D10(&v36, a2, v20, v21, v22);
      }
      else
      {
        v36 = ~v36;
        sub_AAD510(&v36);
      }
      v23 = v37;
      v30 = v36;
      v31 = v37;
      v24 = 1LL << ((unsigned __int8)v29 - 1);
      if ( v29 > 0x40 )
      {
        *(_QWORD *)(v28 + 8LL * ((v29 - 1) >> 6)) |= v24;
        v23 = v31;
      }
      else
      {
        v28 |= v24;
      }
      v25 = ~(1LL << ((unsigned __int8)v23 - 1));
      if ( v23 > 0x40 )
        *(_QWORD *)(v30 + 8LL * ((v23 - 1) >> 6)) &= v25;
      else
        v30 &= v25;
      sub_9865C0((__int64)&v32, (__int64)&v30);
      sub_C46A40(&v32, 1);
      v26 = v33;
      v33 = 0;
      v35 = v26;
      v34 = v32;
      sub_9865C0((__int64)&v36, (__int64)&v28);
      sub_AADC30(a1, (__int64)&v36, (__int64 *)&v34);
      sub_969240((__int64 *)&v36);
      sub_969240((__int64 *)&v34);
      sub_969240((__int64 *)&v32);
      sub_969240((__int64 *)&v30);
      sub_969240(&v28);
      return a1;
    }
    if ( (*(_QWORD *)(v12 + 8LL * ((v6 - 1) >> 6)) & v11) == 0 )
      goto LABEL_45;
    goto LABEL_35;
  }
LABEL_38:
  v37 = v6;
  if ( v6 <= 0x40 )
  {
    v12 = *(_QWORD *)a2;
    goto LABEL_11;
  }
LABEL_36:
  sub_C43780(&v36, a2);
  v6 = v37;
  if ( v37 <= 0x40 )
  {
LABEL_12:
    v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v36;
    if ( !v6 )
      v13 = 0;
    goto LABEL_14;
  }
  sub_C43D10(&v36, a2, v17, v18, v19);
  v6 = v37;
  v13 = v36;
LABEL_14:
  v35 = v6;
  v34 = v13;
  sub_C46A40(&v34, 1);
  v14 = v35;
  v35 = 0;
  v37 = v14;
  v36 = v34;
  v33 = *(_DWORD *)(a2 + 24);
  if ( v33 > 0x40 )
    sub_C43780(&v32, v3);
  else
    v32 = *(_QWORD *)(a2 + 16);
  sub_AADC30(a1, (__int64)&v32, (__int64 *)&v36);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  return a1;
}
