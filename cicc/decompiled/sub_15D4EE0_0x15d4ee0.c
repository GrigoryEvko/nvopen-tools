// Function: sub_15D4EE0
// Address: 0x15d4ee0
//
__int64 *__fastcall sub_15D4EE0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 *result; // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // rdi
  char v15; // al
  char *v16; // r8
  __int64 v17; // r8
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  char *v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 *v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h] BYREF
  __int64 v31; // [rsp+40h] [rbp-40h] BYREF
  char *v32[7]; // [rsp+48h] [rbp-38h] BYREF

  v20 = a1 + 24;
  *(_QWORD *)(sub_15D4720(a1 + 24, (__int64 *)(*(_QWORD *)a1 + 8LL)) + 32) = *a3;
  result = *(__int64 **)a1;
  v28 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  if ( v28 != 1 )
  {
    v5 = a2 + 48;
    v6 = 1;
    while ( 1 )
    {
      v29 = result[v6];
      result = sub_15CFF10(v5, &v29);
      if ( result[1] )
        goto LABEL_3;
      v31 = v29;
      v7 = sub_15CE6E0(v20, &v31, v32);
      v8 = 0;
      if ( v7 && v32[0] != (char *)(*(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48)) )
        v8 = *((_QWORD *)v32[0] + 4);
      v30 = v8;
      v21 = v8;
      v9 = sub_15CC960(a2, v8);
      if ( !v9 )
      {
        v31 = v21;
        v15 = sub_15CE6E0(v20, &v31, v32);
        v16 = 0;
        if ( v15 && v32[0] != (char *)(*(_QWORD *)(a1 + 32) + 72LL * *(unsigned int *)(a1 + 48)) )
          v16 = (char *)*((_QWORD *)v32[0] + 4);
        v24 = sub_15D0230(a1, v16, a2);
        sub_15CC0B0(&v31, v30, (__int64)v24);
        v32[0] = (char *)v31;
        sub_15CE4A0((__int64)(v24 + 24), v32);
        v17 = v31;
        v31 = 0;
        v25 = v17;
        v18 = sub_15CFF10(v5, &v30);
        v9 = v25;
        v19 = v18[1];
        v26 = v18;
        v18[1] = v9;
        if ( v19 )
        {
          sub_15CBC60(v19);
          v9 = v26[1];
        }
        if ( v31 )
        {
          v27 = v9;
          sub_15CBC60(v31);
          v9 = v27;
        }
      }
      v22 = v9;
      sub_15CC0B0(&v31, v29, v9);
      v32[0] = (char *)v31;
      sub_15CE4A0(v22 + 24, v32);
      v10 = v31;
      v31 = 0;
      v23 = v10;
      result = sub_15CFF10(v5, &v29);
      v11 = result[1];
      result[1] = v23;
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 24);
        if ( v12 )
          j_j___libc_free_0(v12, *(_QWORD *)(v11 + 40) - v12);
        result = (__int64 *)j_j___libc_free_0(v11, 56);
      }
      v13 = v31;
      if ( !v31 )
      {
LABEL_3:
        if ( v28 == ++v6 )
          return result;
      }
      else
      {
        v14 = *(_QWORD *)(v31 + 24);
        if ( v14 )
          j_j___libc_free_0(v14, *(_QWORD *)(v31 + 40) - v14);
        ++v6;
        result = (__int64 *)j_j___libc_free_0(v13, 56);
        if ( v28 == v6 )
          return result;
      }
      result = *(__int64 **)a1;
    }
  }
  return result;
}
