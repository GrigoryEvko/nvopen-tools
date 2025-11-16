// Function: sub_1005FE0
// Address: 0x1005fe0
//
const void **__fastcall sub_1005FE0(const void ***a1, const void **a2, const void **a3, __int64 a4, char a5)
{
  bool v11; // zf
  __int64 v12; // rdi
  _BYTE *v13; // rsi
  unsigned int v14; // edx
  unsigned __int64 v15; // r8
  void *v16; // r8
  bool v17; // cc
  bool v18; // al
  unsigned int v19; // edx
  unsigned __int64 v20; // r8
  const void *v21; // r8
  bool v22; // al
  __int64 v23; // rdx
  _BYTE *v24; // rax
  unsigned int v25; // [rsp+0h] [rbp-90h]
  const void *v26; // [rsp+0h] [rbp-90h]
  void *v27; // [rsp+8h] [rbp-88h]
  bool v28; // [rsp+8h] [rbp-88h]
  unsigned int v29; // [rsp+8h] [rbp-88h]
  bool v30; // [rsp+8h] [rbp-88h]
  const void **v31; // [rsp+18h] [rbp-78h] BYREF
  const void *v32; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-68h]
  void *v34; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-58h]
  const void **v36; // [rsp+40h] [rbp-50h] BYREF
  const void ***v37; // [rsp+48h] [rbp-48h] BYREF
  char v38; // [rsp+50h] [rbp-40h]

  if ( a2 != a3 || *(_BYTE *)a1 != 57 || a2 != *(a1 - 8) )
    goto LABEL_2;
  v12 = (__int64)*(a1 - 4);
  if ( *(_BYTE *)v12 == 17 )
  {
    v13 = (_BYTE *)(v12 + 24);
    v31 = (const void **)(v12 + 24);
    goto LABEL_23;
  }
  v23 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17;
  if ( (unsigned int)v23 > 1 || *(_BYTE *)v12 > 0x15u || (v24 = sub_AD7630(v12, 0, v23)) == 0 || *v24 != 17 )
  {
LABEL_2:
    if ( a3 != (const void **)a1 )
      goto LABEL_3;
LABEL_37:
    v11 = *(_BYTE *)a2 == 57;
    v36 = a3;
    v37 = &v31;
    v38 = 0;
    if ( !v11 || a3 != *(a2 - 8) || !(unsigned __int8)sub_991580((__int64)&v37, (__int64)*(a2 - 4)) )
      goto LABEL_3;
    v19 = *((_DWORD *)v31 + 2);
    v33 = v19;
    if ( v19 > 0x40 )
    {
      sub_C43780((__int64)&v32, v31);
      v19 = v33;
      if ( v33 > 0x40 )
      {
        sub_C43D10((__int64)&v32);
        v21 = v32;
        v19 = v33;
        goto LABEL_45;
      }
      v20 = (unsigned __int64)v32;
    }
    else
    {
      v20 = (unsigned __int64)*v31;
    }
    v21 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v19) & ~v20);
    if ( !v19 )
      v21 = 0;
    v32 = v21;
LABEL_45:
    v17 = *(_DWORD *)(a4 + 8) <= 0x40u;
    v35 = v19;
    v34 = (void *)v21;
    v33 = 0;
    if ( v17 )
    {
      v22 = *(_QWORD *)a4 == (_QWORD)v21;
    }
    else
    {
      v26 = v21;
      v29 = v19;
      v22 = sub_C43C50(a4, (const void **)&v34);
      v21 = v26;
      v19 = v29;
    }
    if ( v19 > 0x40 )
    {
      if ( v21 )
      {
        v30 = v22;
        j_j___libc_free_0_0(v21);
        v22 = v30;
        if ( v33 > 0x40 )
        {
          if ( v32 )
          {
            j_j___libc_free_0_0(v32);
            v22 = v30;
          }
        }
      }
    }
    if ( v22 )
      goto LABEL_53;
    goto LABEL_3;
  }
  v13 = v24 + 24;
  v31 = (const void **)(v24 + 24);
LABEL_23:
  v14 = *((_DWORD *)v13 + 2);
  v35 = v14;
  if ( v14 > 0x40 )
  {
    sub_C43780((__int64)&v34, (const void **)v13);
    v14 = v35;
    if ( v35 > 0x40 )
    {
      sub_C43D10((__int64)&v34);
      v14 = v35;
      v16 = v34;
      goto LABEL_28;
    }
    v15 = (unsigned __int64)v34;
  }
  else
  {
    v15 = *(_QWORD *)v13;
  }
  v16 = (void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v15);
  if ( !v14 )
    v16 = 0;
  v34 = v16;
LABEL_28:
  v17 = *(_DWORD *)(a4 + 8) <= 0x40u;
  LODWORD(v37) = v14;
  v36 = (const void **)v16;
  v35 = 0;
  if ( v17 )
  {
    v18 = *(_QWORD *)a4 == (_QWORD)v16;
  }
  else
  {
    v25 = v14;
    v27 = v16;
    v18 = sub_C43C50(a4, (const void **)&v36);
    v14 = v25;
    v16 = v27;
  }
  if ( v14 > 0x40 )
  {
    if ( v16 )
    {
      v28 = v18;
      j_j___libc_free_0_0(v16);
      v18 = v28;
      if ( v35 > 0x40 )
      {
        if ( v34 )
        {
          j_j___libc_free_0_0(v34);
          v18 = v28;
        }
      }
    }
  }
  if ( v18 )
  {
LABEL_53:
    if ( !a5 )
      return (const void **)a1;
    return a2;
  }
  if ( a3 == (const void **)a1 )
    goto LABEL_37;
LABEL_3:
  if ( *(_DWORD *)(a4 + 8) > 0x40u )
  {
    if ( (unsigned int)sub_C44630(a4) != 1 )
      return 0;
  }
  else if ( !*(_QWORD *)a4 || (*(_QWORD *)a4 & (*(_QWORD *)a4 - 1LL)) != 0 )
  {
    return 0;
  }
  if ( a2 == a3 )
  {
    v11 = *(_BYTE *)a1 == 58;
    v36 = a2;
    v37 = &v31;
    v38 = 0;
    if ( v11 && a2 == *(a1 - 8) && (unsigned __int8)sub_991580((__int64)&v37, (__int64)*(a1 - 4)) )
    {
      if ( *(_DWORD *)(a4 + 8) <= 0x40u )
      {
        if ( *(const void **)a4 == *v31 )
          goto LABEL_60;
      }
      else if ( sub_C43C50(a4, v31) )
      {
LABEL_60:
        if ( !a5 )
          return a2;
        if ( (*((_BYTE *)a1 + 1) & 2) == 0 )
          return (const void **)a1;
        return 0;
      }
    }
  }
  if ( a3 != (const void **)a1 )
    return 0;
  v11 = *(_BYTE *)a2 == 58;
  v36 = (const void **)a1;
  v37 = &v31;
  v38 = 0;
  if ( !v11 || a1 != *(a2 - 8) || !(unsigned __int8)sub_991580((__int64)&v37, (__int64)*(a2 - 4)) )
    return 0;
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
  {
    if ( *(const void **)a4 != *v31 )
      return 0;
  }
  else if ( !sub_C43C50(a4, v31) )
  {
    return 0;
  }
  if ( !a5 )
  {
    if ( (*((_BYTE *)a2 + 1) & 2) == 0 )
      return a2;
    return 0;
  }
  return (const void **)a1;
}
