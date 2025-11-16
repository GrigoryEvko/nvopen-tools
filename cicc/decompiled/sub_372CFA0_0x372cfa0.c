// Function: sub_372CFA0
// Address: 0x372cfa0
//
__int64 __fastcall sub_372CFA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // r9
  __int64 v8; // r13
  unsigned int v9; // eax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  __int64 result; // rax
  char v17; // r8
  __m128i v18; // [rsp+0h] [rbp-30h] BYREF

  v18.m128i_i64[0] = a2;
  v18.m128i_i64[1] = a3;
  v8 = sub_372CB80(a1, &v18);
  v9 = *(_DWORD *)(v8 + 8);
  if ( !v9 )
  {
    v10 = 0;
LABEL_13:
    v12 = *(unsigned int *)(v8 + 12);
    if ( v12 > v10 )
    {
      v13 = (unsigned __int64 *)(*(_QWORD *)v8 + 16 * v10);
      if ( !v13 )
        goto LABEL_7;
      goto LABEL_5;
    }
    goto LABEL_16;
  }
  v10 = v9;
  v11 = (_QWORD *)(*(_QWORD *)v8 + 16LL * v9 - 16);
  if ( (*v11 & 4) != 0 || v11[1] != -1 )
  {
    v12 = *(unsigned int *)(v8 + 12);
    if ( v9 < v12 )
    {
      v13 = (unsigned __int64 *)(*(_QWORD *)v8 + 16LL * v9);
LABEL_5:
      v14 = a4 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_6:
      *v13 = v14;
      v13[1] = -1;
      v9 = *(_DWORD *)(v8 + 8);
LABEL_7:
      v15 = v9 + 1;
      *(_DWORD *)(v8 + 8) = v15;
      *a5 = v15 - 1;
      return 1;
    }
LABEL_16:
    v14 = a4 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v12 < v10 + 1 )
    {
      sub_C8D5F0(v8, (const void *)(v8 + 16), v10 + 1, 0x10u, v10 + 1, v7);
      v10 = *(unsigned int *)(v8 + 8);
    }
    v13 = (unsigned __int64 *)(*(_QWORD *)v8 + 16 * v10);
    goto LABEL_6;
  }
  v17 = sub_2E891D0(*v11 & 0xFFFFFFFFFFFFFFF8LL, a4);
  result = 0;
  if ( !v17 )
  {
    v10 = *(unsigned int *)(v8 + 8);
    v9 = *(_DWORD *)(v8 + 8);
    goto LABEL_13;
  }
  return result;
}
