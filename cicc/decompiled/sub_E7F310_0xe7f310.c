// Function: sub_E7F310
// Address: 0xe7f310
//
__int64 __fastcall sub_E7F310(__int64 a1, int a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // r12
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __m128i v10; // xmm0
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rbx
  int v15; // eax
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  _BYTE *v18; // r15
  size_t v19; // r12
  _OWORD *v20; // rdi
  __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  char *v24; // r12
  size_t v25[2]; // [rsp+0h] [rbp-80h] BYREF
  __m128i v26; // [rsp+10h] [rbp-70h] BYREF
  _DWORD v27[4]; // [rsp+20h] [rbp-60h] BYREF
  _OWORD *v28; // [rsp+30h] [rbp-50h]
  __int64 v29; // [rsp+38h] [rbp-48h]
  _OWORD v30[4]; // [rsp+40h] [rbp-40h] BYREF

  v7 = (char *)v27;
  v27[2] = a3;
  v8 = *(unsigned int *)(a1 + 3536);
  v26.m128i_i8[0] = 0;
  v9 = *(_QWORD *)(a1 + 3528);
  v10 = _mm_load_si128(&v26);
  v11 = *(unsigned int *)(a1 + 3540);
  v27[1] = a2;
  v12 = v8 + 1;
  v27[0] = 1;
  v13 = v8;
  v28 = v30;
  v29 = 0;
  v30[0] = v10;
  if ( v8 + 1 > v11 )
  {
    v23 = a1 + 3528;
    if ( v9 > (unsigned __int64)v27 || (unsigned __int64)v27 >= v9 + 48 * v8 )
    {
      sub_E7F1B0(v23, v12, v8, v9, a5, a6);
      v8 = *(unsigned int *)(a1 + 3536);
      v9 = *(_QWORD *)(a1 + 3528);
      v13 = *(_DWORD *)(a1 + 3536);
    }
    else
    {
      v24 = (char *)v27 - v9;
      sub_E7F1B0(v23, v12, v8, v9, a5, a6);
      v9 = *(_QWORD *)(a1 + 3528);
      v8 = *(unsigned int *)(a1 + 3536);
      v7 = &v24[v9];
      v13 = *(_DWORD *)(a1 + 3536);
    }
  }
  v14 = v9 + 48 * v8;
  if ( v14 )
  {
    v15 = *((_DWORD *)v7 + 2);
    v16 = *(_QWORD *)v7;
    v17 = (_BYTE *)(v14 + 32);
    *(_QWORD *)(v14 + 16) = v14 + 32;
    *(_DWORD *)(v14 + 8) = v15;
    *(_QWORD *)v14 = v16;
    v18 = (_BYTE *)*((_QWORD *)v7 + 2);
    v19 = *((_QWORD *)v7 + 3);
    if ( &v18[v19] && !v18 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v25[0] = v19;
    if ( v19 > 0xF )
    {
      v22 = sub_22409D0(v14 + 16, v25, 0);
      *(_QWORD *)(v14 + 16) = v22;
      v17 = (_BYTE *)v22;
      *(_QWORD *)(v14 + 32) = v25[0];
    }
    else
    {
      if ( v19 == 1 )
      {
        *(_BYTE *)(v14 + 32) = *v18;
LABEL_8:
        *(_QWORD *)(v14 + 24) = v19;
        v17[v19] = 0;
        v13 = *(_DWORD *)(a1 + 3536);
        goto LABEL_9;
      }
      if ( !v19 )
        goto LABEL_8;
    }
    memcpy(v17, v18, v19);
    v19 = v25[0];
    v17 = *(_BYTE **)(v14 + 16);
    goto LABEL_8;
  }
LABEL_9:
  v20 = v28;
  result = (unsigned int)(v13 + 1);
  *(_DWORD *)(a1 + 3536) = result;
  if ( v20 != v30 )
    return j_j___libc_free_0(v20, *(_QWORD *)&v30[0] + 1LL);
  return result;
}
