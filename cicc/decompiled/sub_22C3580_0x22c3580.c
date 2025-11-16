// Function: sub_22C3580
// Address: 0x22c3580
//
__int64 __fastcall sub_22C3580(__int64 a1, _BYTE *a2, _BYTE *a3, int a4)
{
  char v7; // al
  char v8; // al
  unsigned int v9; // eax
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  bool v14; // cc
  unsigned __int64 v15; // rdi
  __int64 result; // rax
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  const void **v20; // rsi
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  const void **v23; // [rsp+8h] [rbp-78h] BYREF
  _BYTE *v24; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-68h]
  _BYTE *v26; // [rsp+20h] [rbp-60h] BYREF
  const void ***v27; // [rsp+28h] [rbp-58h] BYREF
  char v28; // [rsp+30h] [rbp-50h]
  _BYTE *v29; // [rsp+38h] [rbp-48h]
  const void ***v30; // [rsp+40h] [rbp-40h] BYREF
  char v31; // [rsp+48h] [rbp-38h]

  v7 = *a2;
  v26 = a3;
  v27 = &v23;
  v28 = 0;
  v29 = a3;
  v30 = &v23;
  v31 = 0;
  if ( v7 == 42 )
  {
    if ( a3 != *((_BYTE **)a2 - 8) )
      goto LABEL_3;
    if ( (unsigned __int8)sub_991580((__int64)&v27, *((_QWORD *)a2 - 4)) )
      goto LABEL_36;
    v7 = *a2;
  }
  if ( v7 == 58
    && (a2[1] & 2) != 0
    && *((_BYTE **)a2 - 8) == v29
    && (unsigned __int8)sub_991580((__int64)&v30, *((_QWORD *)a2 - 4)) )
  {
LABEL_36:
    v20 = v23;
    if ( *(_DWORD *)(a1 + 8) <= 0x40u && *((_DWORD *)v23 + 2) <= 0x40u )
    {
      *(_QWORD *)a1 = *v23;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)v20 + 2);
      return 1;
    }
    else
    {
      sub_C43990(a1, (__int64)v23);
      return 1;
    }
  }
LABEL_3:
  v8 = *a3;
  v26 = a2;
  v27 = &v23;
  v28 = 0;
  v29 = a2;
  v30 = &v23;
  v31 = 0;
  if ( v8 == 42 )
  {
    v17 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( a2 != v17 || !v17 )
    {
LABEL_21:
      if ( *a2 == 58 )
      {
        v21 = (_BYTE *)*((_QWORD *)a2 - 8);
        if ( a3 == v21 && v21 || (v22 = (_BYTE *)*((_QWORD *)a2 - 4), a3 == v22) && v22 )
        {
          result = 1;
          if ( (unsigned int)(a4 - 36) <= 1 )
            return result;
        }
      }
      else if ( *a2 == 57 )
      {
        if ( (v18 = (_BYTE *)*((_QWORD *)a2 - 8), a3 == v18) && v18
          || (v19 = (_BYTE *)*((_QWORD *)a2 - 4), a3 == v19) && v19 )
        {
          result = 1;
          if ( (unsigned int)(a4 - 34) <= 1 )
            return result;
        }
      }
      return 0;
    }
    if ( (unsigned __int8)sub_991580((__int64)&v27, *((_QWORD *)a3 - 4)) )
      goto LABEL_8;
    v8 = *a3;
  }
  if ( v8 != 58
    || (a3[1] & 2) == 0
    || *((_BYTE **)a3 - 8) != v29
    || !(unsigned __int8)sub_991580((__int64)&v30, *((_QWORD *)a3 - 4)) )
  {
    goto LABEL_21;
  }
LABEL_8:
  v9 = *((_DWORD *)v23 + 2);
  v25 = v9;
  if ( v9 <= 0x40 )
  {
    v10 = (__int64)*v23;
LABEL_10:
    v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & ~v10;
    if ( !v9 )
      v11 = 0;
    v24 = (_BYTE *)v11;
    goto LABEL_13;
  }
  sub_C43780((__int64)&v24, v23);
  v9 = v25;
  if ( v25 <= 0x40 )
  {
    v10 = (__int64)v24;
    goto LABEL_10;
  }
  sub_C43D10((__int64)&v24);
LABEL_13:
  sub_C46250((__int64)&v24);
  v12 = v25;
  v13 = (__int64)v24;
  v25 = 0;
  v14 = *(_DWORD *)(a1 + 8) <= 0x40u;
  LODWORD(v27) = v12;
  v26 = v24;
  if ( !v14 )
  {
    v15 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      j_j___libc_free_0_0(v15);
      v13 = (__int64)v26;
      v12 = (unsigned int)v27;
    }
  }
  *(_DWORD *)(a1 + 8) = v12;
  *(_QWORD *)a1 = v13;
  LODWORD(v27) = 0;
  sub_969240((__int64 *)&v26);
  sub_969240((__int64 *)&v24);
  return 1;
}
