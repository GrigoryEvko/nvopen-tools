// Function: sub_14C00B0
// Address: 0x14c00b0
//
__int64 __fastcall sub_14C00B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rax
  unsigned int v9; // r14d
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  char v13; // al
  __int64 v14; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-D8h]
  __int64 v16[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v17; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-B8h]
  __int64 v19[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v20[5]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v21; // [rsp+78h] [rbp-78h] BYREF
  __int64 v22; // [rsp+80h] [rbp-70h]
  _BYTE v23[48]; // [rsp+88h] [rbp-68h] BYREF
  int v24; // [rsp+B8h] [rbp-38h]

  if ( !a5 || !*(_QWORD *)(a5 + 40) )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x17u || (a5 = a2, !*(_QWORD *)(a2 + 40)) )
    {
      a5 = 0;
      if ( *((_BYTE *)a1 + 16) > 0x17u )
      {
        a5 = a1[5];
        if ( a5 )
          a5 = (__int64)a1;
      }
    }
  }
  v20[0] = a3;
  v20[1] = a4;
  v20[2] = a5;
  v20[3] = a6;
  v20[4] = 0;
  v21 = v23;
  v22 = 0x600000000LL;
  v24 = 0;
  v6 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v6 )
    v7 = *v6;
  else
    v7 = qword_4FBB370[2];
  v8 = (unsigned int)v22;
  if ( v7 < (unsigned __int64)(unsigned int)v22 )
    goto LABEL_25;
  if ( v7 > (unsigned __int64)(unsigned int)v22 )
  {
    if ( v7 > (unsigned __int64)HIDWORD(v22) )
    {
      sub_16CD150(&v21, v23, v7, 8);
      v8 = (unsigned int)v22;
    }
    v11 = &v21[8 * v8];
    for ( i = &v21[8 * v7]; i != v11; ++v11 )
    {
      if ( v11 )
        *v11 = 0;
    }
LABEL_25:
    LODWORD(v22) = v7;
  }
  if ( (__int64 *)a2 == a1 || *a1 != *(_QWORD *)a2 )
  {
LABEL_9:
    v9 = 0;
    goto LABEL_10;
  }
  if ( (unsigned __int8)sub_14C0080((__int64)a1, a2, v20) )
    goto LABEL_28;
  v9 = sub_14C0080(a2, (__int64)a1, v20);
  if ( (_BYTE)v9 )
    goto LABEL_28;
  v13 = *(_BYTE *)(*a1 + 8);
  if ( v13 == 16 )
    v13 = *(_BYTE *)(**(_QWORD **)(*a1 + 16) + 8LL);
  if ( v13 != 11 )
    goto LABEL_9;
  sub_14BCFC0((__int64)&v14, a1, 0, v20);
  sub_14BCFC0((__int64)&v17, (__int64 *)a2, 0, v20);
  if ( v15 <= 0x40 )
  {
    if ( (v19[0] & v14) != 0 )
      goto LABEL_39;
  }
  else if ( (unsigned __int8)sub_16A59B0(&v14, v19) )
  {
    goto LABEL_39;
  }
  if ( v18 <= 0x40 )
  {
    if ( (v16[0] & v17) == 0 )
      goto LABEL_37;
    goto LABEL_39;
  }
  if ( (unsigned __int8)sub_16A59B0(&v17, v16) )
  {
LABEL_39:
    sub_135E100(v19);
    sub_135E100(&v17);
    sub_135E100(v16);
    sub_135E100(&v14);
LABEL_28:
    v9 = 1;
    goto LABEL_10;
  }
LABEL_37:
  sub_135E100(v19);
  sub_135E100(&v17);
  sub_135E100(v16);
  sub_135E100(&v14);
LABEL_10:
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  return v9;
}
