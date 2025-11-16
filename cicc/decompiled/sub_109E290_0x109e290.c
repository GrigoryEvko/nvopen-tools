// Function: sub_109E290
// Address: 0x109e290
//
__int64 __fastcall sub_109E290(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r14
  void *v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // r15

  v2 = (__int64 *)(a1 + 8);
  v4 = sub_C33340();
  v5 = *a2;
  if ( !*(_BYTE *)a1 )
  {
    v6 = (_QWORD *)(a1 + 8);
    if ( v4 != (void *)v5 )
    {
LABEL_3:
      sub_C33EB0(v6, a2);
      goto LABEL_4;
    }
LABEL_13:
    sub_C3C790(v6, (_QWORD **)a2);
    goto LABEL_4;
  }
  if ( *(void **)(a1 + 8) != v4 )
  {
    if ( v4 != (void *)v5 )
    {
      sub_C33E70((__int64 *)(a1 + 8), a2);
      goto LABEL_4;
    }
    if ( a2 == v2 )
      goto LABEL_4;
    sub_C338F0(a1 + 8);
    v8 = *a2;
    goto LABEL_10;
  }
  if ( v4 == (void *)v5 )
  {
    sub_C3C9E0((__int64 *)(a1 + 8), a2);
    goto LABEL_4;
  }
  if ( a2 != v2 )
  {
    v9 = *(_QWORD **)(a1 + 16);
    if ( !v9 )
      goto LABEL_11;
    v10 = &v9[3 * *(v9 - 1)];
    if ( v9 != v10 )
    {
      do
      {
        v10 -= 3;
        sub_91D830(v10);
      }
      while ( *(_QWORD **)(a1 + 16) != v10 );
    }
    j_j_j___libc_free_0_0(v10 - 1);
    v8 = *a2;
LABEL_10:
    if ( (void *)v8 == v4 )
    {
      v6 = (_QWORD *)(a1 + 8);
      goto LABEL_13;
    }
LABEL_11:
    v6 = (_QWORD *)(a1 + 8);
    goto LABEL_3;
  }
LABEL_4:
  *(_WORD *)a1 = 257;
  return 257;
}
