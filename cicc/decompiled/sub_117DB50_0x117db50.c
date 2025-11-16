// Function: sub_117DB50
// Address: 0x117db50
//
char __fastcall sub_117DB50(__int64 a1, _BYTE *a2, char a3)
{
  __int64 v3; // r14
  __int64 v4; // rdi
  _BYTE *v6; // r13
  bool v7; // zf
  char result; // al
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r14d
  __int64 v14; // r15
  int v15; // esi
  __int64 v16; // rcx
  char v17; // [rsp+0h] [rbp-A0h]
  char v18; // [rsp+0h] [rbp-A0h]
  char v19; // [rsp+0h] [rbp-A0h]
  char v20; // [rsp+0h] [rbp-A0h]
  char v21; // [rsp+0h] [rbp-A0h]
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+0h] [rbp-A0h]
  const void **v24; // [rsp+10h] [rbp-90h] BYREF
  int v25; // [rsp+18h] [rbp-88h] BYREF
  char v26; // [rsp+1Ch] [rbp-84h]
  __int64 v27; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-68h]
  __int64 v31; // [rsp+40h] [rbp-60h]
  unsigned int v32; // [rsp+48h] [rbp-58h]
  __int64 v33; // [rsp+50h] [rbp-50h] BYREF
  __int64 v34; // [rsp+58h] [rbp-48h]
  const void ***v35; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v36; // [rsp+68h] [rbp-38h]

  v26 = 0;
  v3 = *(_QWORD *)(a1 - 64);
  v25 = 42;
  if ( (*(_BYTE *)(a1 + 1) & 2) == 0 )
    return 0;
  v4 = *(_QWORD *)(a1 - 32);
  v6 = (_BYTE *)(v4 + 24);
  if ( *(_BYTE *)v4 != 17 )
  {
    v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
    if ( (unsigned int)v9 > 1 )
      return 0;
    if ( *(_BYTE *)v4 > 0x15u )
      return 0;
    v10 = sub_AD7630(v4, 0, v9);
    if ( !v10 || *v10 != 17 )
      return 0;
    v6 = v10 + 24;
  }
  v7 = *a2 == 82;
  v34 = v3;
  v33 = (__int64)&v25;
  v35 = &v24;
  LOBYTE(v36) = 1;
  if ( !v7 || v3 != *((_QWORD *)a2 - 8) || !(unsigned __int8)sub_991580((__int64)&v35, *((_QWORD *)a2 - 4)) )
    return 0;
  if ( v33 )
  {
    v11 = sub_B53900((__int64)a2);
    v12 = v33;
    *(_DWORD *)v33 = v11;
    *(_BYTE *)(v12 + 4) = BYTE4(v11);
  }
  v13 = *((_DWORD *)v6 + 2);
  v14 = 1LL << ((unsigned __int8)v13 - 1);
  if ( v13 > 0x40 )
  {
    v16 = (v13 - 1) >> 6;
    if ( (*(_QWORD *)(*(_QWORD *)v6 + 8 * v16) & v14) != 0 )
    {
      v22 = 8 * v16;
      LODWORD(v34) = *((_DWORD *)v6 + 2);
      sub_C43690((__int64)&v33, 0, 0);
      if ( (unsigned int)v34 <= 0x40 )
        v33 |= v14;
      else
        *(_QWORD *)(v33 + v22) |= v14;
      v28 = v13;
      sub_C43690((__int64)&v27, 0, 0);
      goto LABEL_19;
    }
    v23 = 8 * v16;
    LODWORD(v34) = *((_DWORD *)v6 + 2);
    sub_C43690((__int64)&v33, 0, 0);
    v28 = v13;
    sub_C43690((__int64)&v27, 0, 0);
    if ( v28 > 0x40 )
    {
      *(_QWORD *)(v27 + v23) |= v14;
      goto LABEL_19;
    }
  }
  else
  {
    if ( (*(_QWORD *)v6 & v14) != 0 )
    {
      LODWORD(v34) = *((_DWORD *)v6 + 2);
      v33 = 1LL << ((unsigned __int8)v13 - 1);
      v28 = v13;
      v27 = 0;
      goto LABEL_19;
    }
    LODWORD(v34) = *((_DWORD *)v6 + 2);
    v33 = 0;
    v28 = v13;
    v27 = 0;
  }
  v27 |= v14;
LABEL_19:
  sub_AADC30((__int64)&v29, (__int64)&v27, &v33);
  sub_969240(&v27);
  sub_969240(&v33);
  v28 = *((_DWORD *)v24 + 2);
  if ( v28 > 0x40 )
    sub_C43780((__int64)&v27, v24);
  else
    v27 = (__int64)*v24;
  sub_AADBC0((__int64)&v33, &v27);
  if ( a3 )
    v15 = v25;
  else
    v15 = sub_B52870(v25);
  result = sub_ABB410(&v29, v15, &v33);
  if ( v36 > 0x40 && v35 )
  {
    v17 = result;
    j_j___libc_free_0_0(v35);
    result = v17;
  }
  if ( (unsigned int)v34 > 0x40 && v33 )
  {
    v18 = result;
    j_j___libc_free_0_0(v33);
    result = v18;
  }
  if ( v28 > 0x40 && v27 )
  {
    v19 = result;
    j_j___libc_free_0_0(v27);
    result = v19;
  }
  if ( v32 > 0x40 && v31 )
  {
    v20 = result;
    j_j___libc_free_0_0(v31);
    result = v20;
  }
  if ( v30 > 0x40 )
  {
    if ( v29 )
    {
      v21 = result;
      j_j___libc_free_0_0(v29);
      return v21;
    }
  }
  return result;
}
