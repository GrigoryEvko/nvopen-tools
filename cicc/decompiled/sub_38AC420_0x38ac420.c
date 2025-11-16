// Function: sub_38AC420
// Address: 0x38ac420
//
__int64 __fastcall sub_38AC420(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  char v6; // r14
  int v7; // eax
  unsigned __int64 v8; // r15
  bool v9; // zf
  const char *v10; // rax
  int v12; // eax
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int16 v15; // ax
  __int16 v16; // ax
  int v17; // eax
  char v19; // [rsp+1Fh] [rbp-81h]
  char v20; // [rsp+2Fh] [rbp-71h] BYREF
  unsigned int v21; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+34h] [rbp-6Ch] BYREF
  __int64 v23; // [rsp+38h] [rbp-68h] BYREF
  unsigned __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v25; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v26[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v27; // [rsp+60h] [rbp-40h]

  v6 = 0;
  v7 = *(_DWORD *)(a1 + 64);
  v23 = 0;
  v24 = 0;
  v21 = 0;
  v22 = 0;
  v25 = 0;
  if ( v7 == 150 )
  {
    v6 = 1;
    v7 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v7;
  }
  v19 = 0;
  if ( v7 == 194 )
  {
    v19 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v8 = *(_QWORD *)(a1 + 56);
  v26[0] = "expected type";
  v27 = 259;
  if ( (unsigned __int8)sub_3891B00(a1, (__int64 *)&v25, (__int64)v26, 0) )
    return 1;
  if ( *((_BYTE *)v25 + 8) == 12 || !sub_1643F60((__int64)v25) )
  {
    HIBYTE(v27) = 1;
    v10 = "invalid type for alloca";
    goto LABEL_14;
  }
  v9 = *(_DWORD *)(a1 + 64) == 4;
  v20 = 0;
  if ( !v9 )
    goto LABEL_9;
  v12 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v12;
  switch ( v12 )
  {
    case 88:
      if ( (unsigned __int8)sub_388C5A0(a1, &v21) || (unsigned __int8)sub_388CBF0(a1, &v22, &v24, &v20) )
        return 1;
      goto LABEL_9;
    case 89:
      v24 = *(_QWORD *)(a1 + 56);
      if ( (unsigned __int8)sub_388BF60(a1, &v22) )
        return 1;
LABEL_9:
      v8 = 0;
      goto LABEL_10;
    case 376:
      v20 = 1;
      goto LABEL_9;
  }
  v8 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v23, a3, a4, a5, a6) )
    return 1;
  if ( *(_DWORD *)(a1 + 64) == 4 )
  {
    v17 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v17;
    if ( v17 == 88 )
    {
      if ( !(unsigned __int8)sub_388C5A0(a1, &v21) && !(unsigned __int8)sub_388CBF0(a1, &v22, &v24, &v20) )
        goto LABEL_10;
      return 1;
    }
    if ( v17 != 89 )
    {
      if ( v17 == 376 )
        v20 = 1;
      goto LABEL_10;
    }
    v24 = *(_QWORD *)(a1 + 56);
    if ( (unsigned __int8)sub_388BF60(a1, &v22) )
      return 1;
  }
LABEL_10:
  if ( v23 && *(_BYTE *)(*(_QWORD *)v23 + 8LL) != 11 )
  {
    HIBYTE(v27) = 1;
    v10 = "element count must have integer type";
LABEL_14:
    v26[0] = v10;
    LOBYTE(v27) = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v8, (__int64)v26);
  }
  v27 = 257;
  v13 = sub_1648A60(64, 1u);
  v14 = v13;
  if ( v13 )
    sub_15F8A50((__int64)v13, v25, v22, v23, v21, (__int64)v26, 0);
  v15 = *((_WORD *)v14 + 9) & 0x7FDF;
  if ( v6 )
    v15 = *((_WORD *)v14 + 9) & 0x7FDF | 0x20;
  v16 = v15 & 0x7FBF;
  if ( v19 )
    v16 |= 0x40u;
  *((_WORD *)v14 + 9) = *((_WORD *)v14 + 9) & 0x8000 | v16;
  v9 = v20 == 0;
  *a2 = v14;
  return 2 * (unsigned int)!v9;
}
