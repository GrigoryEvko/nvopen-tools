// Function: sub_318E5F0
// Address: 0x318e5f0
//
__int64 __fastcall sub_318E5F0(const __m128i *a1, __int64 a2)
{
  __int64 v4; // r12
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // r15
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  char *v22; // r15
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  __m128i v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+20h] [rbp-40h]

  v4 = a1[1].m128i_i64[0];
  v5 = *(_DWORD *)(v4 + 72) == 1;
  v25 = v4;
  v24 = _mm_loadu_si128(a1);
  if ( v5 )
  {
    v10 = sub_22077B0(0x28u);
    v13 = v10;
    if ( v10 )
    {
      *(__m128i *)(v10 + 8) = _mm_loadu_si128(&v24);
      *(_QWORD *)v10 = &unk_4A34750;
      *(_QWORD *)(v10 + 24) = v25;
      *(_QWORD *)(v10 + 32) = sub_318E5D0((__int64)&v24);
    }
    v14 = *(unsigned int *)(v4 + 16);
    v15 = *(unsigned int *)(v4 + 20);
    v23 = v13;
    v16 = (char *)&v23;
    v17 = *(_QWORD *)(v4 + 8);
    v18 = v14 + 1;
    v19 = v14;
    if ( v14 + 1 > v15 )
    {
      v21 = v4 + 8;
      if ( v17 > (unsigned __int64)&v23 || (unsigned __int64)&v23 >= v17 + 8 * v14 )
      {
        sub_31878D0(v21, v18, v14, v17, v11, v12);
        v14 = *(unsigned int *)(v4 + 16);
        v17 = *(_QWORD *)(v4 + 8);
        v19 = *(_DWORD *)(v4 + 16);
      }
      else
      {
        v22 = (char *)&v23 - v17;
        sub_31878D0(v21, v18, v14, v17, v11, v12);
        v17 = *(_QWORD *)(v4 + 8);
        v14 = *(unsigned int *)(v4 + 16);
        v16 = &v22[v17];
        v19 = *(_DWORD *)(v4 + 16);
      }
    }
    v20 = (_QWORD *)(v17 + 8 * v14);
    if ( v20 )
    {
      *v20 = *(_QWORD *)v16;
      *(_QWORD *)v16 = 0;
      v13 = v23;
      v19 = *(_DWORD *)(v4 + 16);
    }
    *(_DWORD *)(v4 + 16) = v19 + 1;
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 24LL))(v13);
  }
  result = a1->m128i_i64[0];
  v7 = *(_QWORD *)(a2 + 16);
  if ( *(_QWORD *)a1->m128i_i64[0] )
  {
    v8 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = v7;
  if ( v7 )
  {
    v9 = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(result + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = result + 8;
    *(_QWORD *)(result + 16) = v7 + 16;
    *(_QWORD *)(v7 + 16) = result;
  }
  return result;
}
