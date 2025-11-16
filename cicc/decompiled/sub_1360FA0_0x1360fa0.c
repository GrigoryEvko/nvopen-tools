// Function: sub_1360FA0
// Address: 0x1360fa0
//
__int64 __fastcall sub_1360FA0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rbx
  _QWORD *v14; // rax
  unsigned int v15; // r13d
  __int64 v16; // r15
  _QWORD *v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+20h] [rbp-70h]
  unsigned int v27; // [rsp+2Ch] [rbp-64h]
  __int64 v28; // [rsp+30h] [rbp-60h]
  __int64 v29; // [rsp+38h] [rbp-58h]
  __m128i v30; // [rsp+40h] [rbp-50h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h]

  result = *(unsigned int *)(a3 + 8);
  if ( (_DWORD)result )
  {
    v28 = 0;
    v5 = a2;
    v24 = 24 * result;
    while ( 1 )
    {
      v9 = *(_QWORD *)a3 + v28;
      v10 = *(_QWORD *)v9;
      v11 = *(_DWORD *)(v9 + 8);
      v27 = *(_DWORD *)(v9 + 12);
      v26 = *(_QWORD *)(v9 + 16);
      v12 = *((unsigned int *)v5 + 2);
      if ( (_DWORD)v12 )
        break;
LABEL_13:
      if ( v26 )
      {
        v30.m128i_i64[0] = v10;
        v30.m128i_i64[1] = __PAIR64__(v27, v11);
        v31 = -v26;
        v19 = *((unsigned int *)v5 + 2);
        if ( (unsigned int)v19 >= *((_DWORD *)v5 + 3) )
        {
          sub_16CD150(v5, a2 + 2, 0, 24);
          v19 = *((unsigned int *)v5 + 2);
        }
        v28 += 24;
        v20 = (__m128i *)(*v5 + 24 * v19);
        v21 = v31;
        *v20 = _mm_loadu_si128(&v30);
        v20[1].m128i_i64[0] = v21;
        result = v28;
        ++*((_DWORD *)v5 + 2);
        if ( v24 == v28 )
          return result;
      }
      else
      {
LABEL_7:
        v28 += 24;
        result = v28;
        if ( v24 == v28 )
          return result;
      }
    }
    v13 = 0;
    v29 = 24 * v12;
    v14 = v5;
    v15 = v11;
    v16 = a1;
    v17 = v14;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_1360E90(v16, *(_QWORD *)(*v17 + v13), v10) )
      {
        v6 = *v17;
        v7 = *v17 + v13;
        if ( *(_DWORD *)(v7 + 8) == v15 && *(_DWORD *)(v7 + 12) == v27 )
          break;
      }
      v13 += 24;
      if ( v29 == v13 )
      {
        v18 = v17;
        a1 = v16;
        v11 = v15;
        v5 = v18;
        goto LABEL_13;
      }
    }
    v8 = *(_QWORD *)(v7 + 16);
    v5 = v17;
    a1 = v16;
    if ( v8 == v26 )
    {
      v22 = *((unsigned int *)v5 + 2);
      v23 = v6 + 24 * v22;
      if ( v23 != v7 + 24 )
      {
        memmove((void *)v7, (const void *)(v7 + 24), v23 - (v7 + 24));
        LODWORD(v22) = *((_DWORD *)v5 + 2);
      }
      *((_DWORD *)v5 + 2) = v22 - 1;
    }
    else
    {
      *(_QWORD *)(v7 + 16) = v8 - v26;
    }
    goto LABEL_7;
  }
  return result;
}
