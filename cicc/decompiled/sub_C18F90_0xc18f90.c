// Function: sub_C18F90
// Address: 0xc18f90
//
_BYTE *__fastcall sub_C18F90(__int64 *a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  char **v11; // r12
  __int64 v12; // rdi
  _BYTE *result; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // esi
  bool v17; // zf
  __int64 v18; // rdx
  int v19; // ecx
  char **v20; // r14
  __int64 v21; // rdx
  int v22; // edi
  char **v23; // [rsp+18h] [rbp-78h]
  char *v24; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+28h] [rbp-68h]
  _BYTE v26[96]; // [rsp+30h] [rbp-60h] BYREF

  v23 = (char **)(a1 + 1);
  if ( sub_C185F0(a5, a2, (__int64)a3) )
  {
    if ( sub_C185F0(a5, (__int64)a3, (__int64)a4) )
    {
      v9 = *a1;
      goto LABEL_4;
    }
    v17 = !sub_C185F0(a5, a2, (__int64)a4);
    v9 = *a1;
    if ( v17 )
    {
      v21 = *(_QWORD *)a2;
      v25 = 0xC00000000LL;
      *a1 = v21;
      *(_QWORD *)a2 = v9;
      v22 = *((_DWORD *)a1 + 4);
      v24 = v26;
      if ( v22 )
        sub_C15E20((__int64)&v24, v23);
      sub_C15E20((__int64)v23, (char **)(a2 + 8));
      v12 = a2 + 8;
      goto LABEL_8;
    }
  }
  else
  {
    if ( sub_C185F0(a5, a2, (__int64)a4) )
    {
      v14 = *a1;
      v15 = *(_QWORD *)a2;
      v25 = 0xC00000000LL;
      *a1 = v15;
      *(_QWORD *)a2 = v14;
      v16 = *((_DWORD *)a1 + 4);
      v24 = v26;
      if ( v16 )
        sub_C15E20((__int64)&v24, v23);
      v11 = (char **)(a2 + 8);
      goto LABEL_7;
    }
    v17 = !sub_C185F0(a5, (__int64)a3, (__int64)a4);
    v9 = *a1;
    if ( v17 )
    {
LABEL_4:
      v10 = *a3;
      v25 = 0xC00000000LL;
      *a1 = v10;
      *a3 = v9;
      LODWORD(v10) = *((_DWORD *)a1 + 4);
      v24 = v26;
      if ( (_DWORD)v10 )
        sub_C15E20((__int64)&v24, v23);
      v11 = (char **)(a3 + 1);
LABEL_7:
      sub_C15E20((__int64)v23, v11);
      v12 = (__int64)v11;
      goto LABEL_8;
    }
  }
  v18 = *a4;
  v25 = 0xC00000000LL;
  *a1 = v18;
  *a4 = v9;
  v19 = *((_DWORD *)a1 + 4);
  v24 = v26;
  if ( v19 )
    sub_C15E20((__int64)&v24, v23);
  v20 = (char **)(a4 + 1);
  sub_C15E20((__int64)v23, v20);
  v12 = (__int64)v20;
LABEL_8:
  sub_C15E20(v12, &v24);
  result = v26;
  if ( v24 != v26 )
    return (_BYTE *)_libc_free(v24, &v24);
  return result;
}
