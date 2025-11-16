// Function: sub_1D323C0
// Address: 0x1d323c0
//
__int64 __fastcall sub_1D323C0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const void **a6,
        double a7,
        double a8,
        double a9)
{
  char v11; // di
  unsigned __int8 *v13; // rax
  __int64 v14; // r8
  const void **v15; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // ebx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int128 v24; // [rsp-10h] [rbp-70h]
  __int128 v25; // [rsp-10h] [rbp-70h]
  unsigned __int8 v26; // [rsp+Fh] [rbp-51h]
  __int64 v27; // [rsp+10h] [rbp-50h] BYREF
  const void **v28; // [rsp+18h] [rbp-48h]
  char v29[8]; // [rsp+20h] [rbp-40h] BYREF
  const void **v30; // [rsp+28h] [rbp-38h]

  v11 = a5;
  v13 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v27 = a5;
  v14 = *v13;
  v15 = (const void **)*((_QWORD *)v13 + 1);
  v28 = a6;
  v29[0] = v14;
  v30 = v15;
  if ( (_BYTE)v14 == v11 )
  {
    if ( (_BYTE)v14 || v15 == a6 )
      goto LABEL_3;
LABEL_10:
    v26 = v14;
    v23 = sub_1F58D40(&v27, a2, a3, a4, v14, a6);
    v20 = v26;
    v19 = v23;
    if ( !v26 )
      goto LABEL_11;
LABEL_6:
    v22 = sub_1D13440(v20);
    goto LABEL_7;
  }
  if ( !v11 )
    goto LABEL_10;
  v19 = sub_1D13440(v11);
  if ( (_BYTE)v20 )
    goto LABEL_6;
LABEL_11:
  v22 = sub_1F58D40(v29, a2, v17, v18, v20, v21);
LABEL_7:
  if ( v22 >= v19 )
  {
LABEL_3:
    *((_QWORD *)&v24 + 1) = a3;
    *(_QWORD *)&v24 = a2;
    return sub_1D309E0(a1, 145, a4, (unsigned int)v27, v28, 0, a7, a8, a9, v24);
  }
  *((_QWORD *)&v25 + 1) = a3;
  *(_QWORD *)&v25 = a2;
  return sub_1D309E0(a1, 143, a4, (unsigned int)v27, v28, 0, a7, a8, a9, v25);
}
