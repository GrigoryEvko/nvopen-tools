// Function: sub_3445D50
// Address: 0x3445d50
//
__int64 __fastcall sub_3445D50(unsigned int **a1, __int64 a2)
{
  unsigned int v2; // r15d
  unsigned int *v4; // r14
  __int64 *v5; // rsi
  void *v6; // rbx
  _QWORD *j; // rbx
  _QWORD *i; // r13
  unsigned int *v9; // [rsp+8h] [rbp-88h]
  unsigned __int8 v10; // [rsp+17h] [rbp-79h]
  __int64 (*v11)(); // [rsp+18h] [rbp-78h]
  void *v12; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v13; // [rsp+28h] [rbp-68h]
  void *v14; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+48h] [rbp-48h]

  v2 = 1;
  if ( *(_DWORD *)(a2 + 24) == 51 )
    return v2;
  v4 = a1[2];
  v11 = *(__int64 (**)())(*(_QWORD *)v4 + 616LL);
  v10 = *(_BYTE *)a1[1];
  v9 = *a1;
  v5 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
  v6 = sub_C33340();
  if ( (void *)*v5 != v6 )
  {
    sub_C33EB0(&v12, v5);
    if ( v12 != v6 )
      goto LABEL_5;
LABEL_14:
    sub_C3CCB0((__int64)&v12);
    if ( v12 != v6 )
      goto LABEL_6;
LABEL_15:
    sub_C3C840(&v14, &v12);
    goto LABEL_7;
  }
  sub_C3C790(&v12, (_QWORD **)v5);
  if ( v12 == v6 )
    goto LABEL_14;
LABEL_5:
  sub_C34440((unsigned __int8 *)&v12);
  if ( v12 == v6 )
    goto LABEL_15;
LABEL_6:
  sub_C338E0((__int64)&v14, (__int64)&v12);
LABEL_7:
  v2 = 0;
  if ( v11 != sub_2FE3170 )
    v2 = ((__int64 (__fastcall *)(unsigned int *, void **, _QWORD, _QWORD, _QWORD))v11)(
           v4,
           &v14,
           *v9,
           *((_QWORD *)v9 + 1),
           v10);
  if ( v6 == v14 )
  {
    if ( v15 )
    {
      for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v14);
  }
  if ( v6 == v12 )
  {
    if ( v13 )
    {
      for ( j = &v13[3 * *(v13 - 1)]; v13 != j; sub_91D830(j) )
        j -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v12);
  }
  return v2;
}
