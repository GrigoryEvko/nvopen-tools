// Function: sub_EDCE10
// Address: 0xedce10
//
unsigned __int64 *__fastcall sub_EDCE10(unsigned __int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 *(__fastcall *v5)(__int64 *, __int64, __int64 **); // rax
  __int64 v6; // rdx
  unsigned __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v16; // [rsp+18h] [rbp-38h]
  __int64 v17; // [rsp+20h] [rbp-30h]

  v4 = *a2;
  v15 = 0;
  v16 = 0;
  v5 = *(__int64 *(__fastcall **)(__int64 *, __int64, __int64 **))(v4 + 32);
  v17 = 0;
  if ( (char *)v5 == (char *)sub_EDCDD0 )
  {
    v6 = a2[60];
    v7 = (unsigned __int64 *)a2[59];
    a2 = (__int64 *)a2[14];
    sub_EDCB10(&v14, (__int64)a2, v7, v6, &v15, 1);
  }
  else
  {
    v5(&v14, (__int64)a2, &v15);
  }
  if ( (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v14 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    a2 = v15;
    sub_ED7D90(a3, v15, 0xCCCCCCCCCCCCCCCDLL * (v16 - v15), v8, v9, v10);
    *a1 = 1;
  }
  v11 = v16;
  v12 = v15;
  if ( v16 != v15 )
  {
    do
    {
      if ( (__int64 *)*v12 != v12 + 3 )
        _libc_free(*v12, a2);
      v12 += 5;
    }
    while ( v11 != v12 );
    v12 = v15;
  }
  if ( v12 )
    j_j___libc_free_0(v12, v17 - (_QWORD)v12);
  return a1;
}
