// Function: sub_EBA760
// Address: 0xeba760
//
__int64 __fastcall sub_EBA760(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  unsigned int v4; // eax
  __int64 v5; // r14
  unsigned int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // rcx
  size_t v9; // r15
  int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdi
  const char *v15; // rax
  __int64 v17; // rdi
  void *s2; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+18h] [rbp-78h]
  __m128i v21; // [rsp+20h] [rbp-70h] BYREF
  size_t n[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v23; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)a1;
  v21 = 0u;
  v3 = sub_ECD7B0(v2);
  v19 = sub_ECD6A0(v3);
  v4 = sub_EB61F0(*(_QWORD *)a1, v21.m128i_i64);
  if ( (_BYTE)v4 )
  {
    v17 = *(_QWORD *)a1;
    n[0] = (size_t)"expected identifier";
    v23 = 259;
    return (unsigned int)sub_ECDA70(v17, v19, n, 0, 0);
  }
  v5 = *(_QWORD *)a1;
  v6 = v4;
  *(__m128i *)n = _mm_loadu_si128(&v21);
  if ( *(_QWORD *)(v5 + 856) )
  {
    if ( v5 + 824 != sub_EA96F0(v5 + 816, (__int64)n) )
      return v6;
  }
  else
  {
    v7 = *(_QWORD *)(v5 + 768);
    v8 = v7 + 16LL * *(unsigned int *)(v5 + 776);
    if ( v7 != v8 )
    {
      v9 = n[1];
      s2 = (void *)n[0];
      while ( 1 )
      {
        if ( *(_QWORD *)(v7 + 8) == v9 )
        {
          if ( !v9 )
            break;
          v20 = v8;
          v10 = memcmp(*(const void **)v7, s2, v9);
          v8 = v20;
          if ( !v10 )
            break;
        }
        v7 += 16;
        if ( v8 == v7 )
          goto LABEL_10;
      }
      if ( v8 != v7 )
        return v6;
    }
  }
LABEL_10:
  v11 = *(_QWORD *)(v5 + 224);
  v23 = 261;
  *(__m128i *)n = v21;
  v12 = sub_E6C460(v11, (const char **)n);
  v13 = **(unsigned int **)(a1 + 8);
  if ( (*(_BYTE *)(v12 + 8) & 2) == 0 || (_DWORD)v13 == 29 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)a1 + 232LL) + 296LL))(
           *(_QWORD *)(*(_QWORD *)a1 + 232LL),
           v12,
           v13) )
    {
      return v6;
    }
    HIBYTE(v23) = 1;
    v14 = *(_QWORD *)a1;
    v15 = "unable to emit symbol attribute";
  }
  else
  {
    HIBYTE(v23) = 1;
    v14 = *(_QWORD *)a1;
    v15 = "non-local symbol required";
  }
  n[0] = (size_t)v15;
  LOBYTE(v23) = 3;
  return (unsigned int)sub_ECDA70(v14, v19, n, 0, 0);
}
