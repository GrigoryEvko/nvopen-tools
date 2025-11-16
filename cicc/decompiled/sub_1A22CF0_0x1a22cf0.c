// Function: sub_1A22CF0
// Address: 0x1a22cf0
//
__int64 __fastcall sub_1A22CF0(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int8 a5, int a6)
{
  unsigned __int64 v7; // r15
  _QWORD *v9; // rdx
  char *v10; // rax
  bool v11; // cf
  __int64 v12; // rbx
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rdx
  int v18; // eax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rbx
  unsigned __int8 v22; // [rsp+8h] [rbp-68h]
  unsigned int v23; // [rsp+Ch] [rbp-64h]
  __m128i v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+20h] [rbp-50h]
  char v26; // [rsp+30h] [rbp-40h]

  if ( a4 )
  {
    v7 = a1[46];
    v23 = *(_DWORD *)(a3 + 8);
    if ( v23 > 0x40 )
    {
      v22 = a5;
      v18 = sub_16A57B0(a3);
      a5 = v22;
      if ( v23 - v18 <= 0x40 )
      {
        v9 = **(_QWORD ***)a3;
        if ( v7 > (unsigned __int64)v9 )
          goto LABEL_4;
      }
    }
    else
    {
      v9 = *(_QWORD **)a3;
      if ( v7 > *(_QWORD *)a3 )
      {
LABEL_4:
        v10 = (char *)v9 + a4;
        v24.m128i_i64[0] = (__int64)v9;
        v11 = v7 - (unsigned __int64)v9 < a4;
        v12 = a1[47];
        if ( !v11 )
          v7 = (unsigned __int64)v10;
        v13 = a1[42] & 0xFFFFFFFFFFFFFFFBLL;
        v24.m128i_i64[1] = v7;
        v14 = v13 | (4LL * a5);
        v15 = *(unsigned int *)(v12 + 16);
        v25 = v14;
        if ( (unsigned int)v15 >= *(_DWORD *)(v12 + 20) )
        {
          sub_16CD150(v12 + 8, (const void *)(v12 + 24), 0, 24, v14, a6);
          v15 = *(unsigned int *)(v12 + 16);
        }
        result = *(_QWORD *)(v12 + 8) + 24 * v15;
        v17 = v25;
        *(__m128i *)result = _mm_loadu_si128(&v24);
        *(_QWORD *)(result + 16) = v17;
        ++*(_DWORD *)(v12 + 16);
        return result;
      }
    }
  }
  result = sub_165A590((__int64)&v24, (__int64)(a1 + 68), a2);
  if ( v26 )
  {
    v21 = a1[47];
    result = *(unsigned int *)(v21 + 224);
    if ( (unsigned int)result >= *(_DWORD *)(v21 + 228) )
    {
      sub_16CD150(v21 + 216, (const void *)(v21 + 232), 0, 8, v19, v20);
      result = *(unsigned int *)(v21 + 224);
    }
    *(_QWORD *)(*(_QWORD *)(v21 + 216) + 8 * result) = a2;
    ++*(_DWORD *)(v21 + 224);
  }
  return result;
}
