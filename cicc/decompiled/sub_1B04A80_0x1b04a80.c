// Function: sub_1B04A80
// Address: 0x1b04a80
//
__int64 __fastcall sub_1B04A80(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 *v8; // rbx
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  void (*v12)(void); // rax
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 *v15; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+28h] [rbp-48h]
  __m128i *v18; // [rsp+38h] [rbp-38h] BYREF

  v15 = *(__int64 **)a1;
  v14 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == v14 )
    return 1;
  while ( 1 )
  {
    v7 = *v15;
    v8 = *(__int64 **)a2;
    v17 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v17 )
      break;
LABEL_22:
    if ( (__int64 *)v14 == ++v15 )
      return 1;
  }
  while ( 1 )
  {
    v10 = *v8;
    if ( v7 == *v8 || *(_BYTE *)(v7 + 16) == 54 && *(_BYTE *)(v10 + 16) == 54 )
      goto LABEL_8;
    sub_13B1040(&v18, a5, v7, v10, 1);
    if ( !v18 )
      goto LABEL_8;
    if ( (*(unsigned __int8 (__fastcall **)(__m128i *))(v18->m128i_i64[0] + 24))(v18) )
      break;
    v11 = v18->m128i_i64[0];
    if ( !a4 )
    {
      if ( ((*(__int64 (__fastcall **)(__m128i *, _QWORD))(v11 + 48))(v18, a3) & 4) != 0 )
        break;
      goto LABEL_5;
    }
    if ( ((*(__int64 (__fastcall **)(__m128i *, _QWORD))(v11 + 48))(v18, a3) & 4) != 0
      && ((*(__int64 (__fastcall **)(__m128i *, _QWORD))(v18->m128i_i64[0] + 48))(v18, a3 + 1) & 1) != 0 )
    {
      break;
    }
LABEL_5:
    if ( v18 )
    {
      v9 = *(__int64 (__fastcall **)(__int64))(v18->m128i_i64[0] + 8);
      if ( v9 == sub_13A31C0 )
      {
        j_j___libc_free_0(v18, 40);
        goto LABEL_8;
      }
      ((void (*)(void))v9)();
      if ( (__int64 *)v17 == ++v8 )
        goto LABEL_22;
    }
    else
    {
LABEL_8:
      if ( (__int64 *)v17 == ++v8 )
        goto LABEL_22;
    }
  }
  if ( v18 )
  {
    v12 = *(void (**)(void))(v18->m128i_i64[0] + 8);
    if ( (char *)v12 == (char *)sub_13A31C0 )
      j_j___libc_free_0(v18, 40);
    else
      v12();
  }
  return 0;
}
