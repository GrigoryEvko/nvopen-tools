// Function: sub_BB50D0
// Address: 0xbb50d0
//
__int64 __fastcall sub_BB50D0(unsigned __int8 **a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  unsigned int v6; // r14d
  unsigned __int8 *v8; // rbx
  bool v9; // zf
  char v10; // [rsp+1Fh] [rbp-61h] BYREF
  unsigned __int64 v11; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-58h]
  __int64 v13; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-48h]
  __int64 v15; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+48h] [rbp-38h]

  sub_C44B10(&v15, a2, *((unsigned int *)*a1 + 2));
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  *(_QWORD *)a2 = v15;
  *(_DWORD *)(a2 + 8) = v16;
  v4 = *((_DWORD *)*a1 + 2);
  v12 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43690(&v11, a3, 0);
  }
  else
  {
    v5 = a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
    if ( !v4 )
      v5 = 0;
    v11 = v5;
  }
  v6 = *a1[1];
  if ( (_BYTE)v6 )
  {
    v10 = 0;
    sub_C4A7C0(&v13, a2, &v11, &v10);
    if ( v10 )
      goto LABEL_17;
    sub_C45F70(&v15, *a1, &v13, &v10);
    v8 = *a1;
    if ( *((_DWORD *)*a1 + 2) > 0x40u && *(_QWORD *)v8 )
      j_j___libc_free_0_0(*(_QWORD *)v8);
    *(_QWORD *)v8 = v15;
    v9 = v10 == 0;
    *((_DWORD *)v8 + 2) = v16;
    if ( !v9 )
    {
LABEL_17:
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      v6 = 0;
    }
    else if ( v14 > 0x40 && v13 )
    {
      j_j___libc_free_0_0(v13);
    }
  }
  else
  {
    sub_C472A0(&v15, a2, &v11);
    sub_C45EE0(*a1, &v15);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    v6 = 1;
  }
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v6;
}
