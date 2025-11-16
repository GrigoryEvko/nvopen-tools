// Function: sub_13A5100
// Address: 0x13a5100
//
__int64 __fastcall sub_13A5100(unsigned __int64 *a1, unsigned int a2, unsigned __int8 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // r12
  bool v8; // zf
  __int64 v9; // rdx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rcx
  __int64 result; // rax
  __int64 v13; // rax
  _QWORD *v14; // r14
  unsigned int v15; // r13d
  __int64 v16; // rax
  size_t v17; // r8
  size_t v18; // rdx
  void *v19; // r12
  unsigned __int64 v20; // r9
  __int64 v21; // rcx
  unsigned __int64 v22; // r8
  __int64 v23; // rbx
  int v24; // r14d
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // [rsp+0h] [rbp-40h]
  size_t v31; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]

  v7 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    if ( a2 > 0x39 )
    {
      v13 = sub_22077B0(24);
      v14 = (_QWORD *)v13;
      if ( v13 )
      {
        *(_DWORD *)(v13 + 16) = a2;
        v15 = (a2 + 63) >> 6;
        *(_QWORD *)v13 = 0;
        *(_QWORD *)(v13 + 8) = 0;
        v16 = malloc(8LL * v15);
        v17 = 8LL * v15;
        v18 = v15;
        v19 = (void *)v16;
        if ( !v16 )
        {
          if ( 8LL * v15 || (v29 = malloc(1u), v18 = v15, v17 = 0, !v29) )
          {
            v31 = v17;
            na = v18;
            sub_16BD1C0("Allocation failed");
            v18 = na;
            v17 = v31;
          }
          else
          {
            v19 = (void *)v29;
          }
        }
        *v14 = v19;
        v14[1] = v18;
        if ( v15 )
          memset(v19, -a3, v17);
        if ( a3 )
          sub_13A4C60((__int64)v14, 0);
      }
      v20 = *a1 >> 58;
      result = -1LL << v20;
      v21 = 0;
      v22 = (*a1 >> 1) & ~(-1LL << v20);
      if ( v20 )
      {
        do
        {
          result = *(_QWORD *)*v14 & ~(1LL << v21);
          if ( ((v22 >> v21) & 1) != 0 )
            result = (1LL << v21) | *(_QWORD *)*v14;
          ++v21;
          *(_QWORD *)*v14 = result;
        }
        while ( v21 != v20 );
      }
      *a1 = (unsigned __int64)v14;
    }
    else
    {
      v8 = a3 == 0;
      v9 = 0;
      if ( !v8 )
        v9 = -1LL << (v7 >> 58);
      v10 = ((unsigned __int64)a2 << 57) | ~(-1LL << (v7 >> 58)) & (v7 >> 1);
      v11 = (2 * v10) >> 58;
      result = ~(-1LL << v11);
      *a1 = 2 * ((v11 << 57) | result & (v9 | result & v10)) + 1;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v7 + 8);
    v24 = a3;
    if ( a2 > (unsigned __int64)(v23 << 6) )
    {
      v25 = (a2 + 63) >> 6;
      if ( v25 < 2 * v23 )
        v25 = 2 * v23;
      v30 = v25;
      n = 8 * v25;
      v26 = (__int64)realloc(*(_QWORD *)v7, 8 * v25, v25, 8 * (int)v25, a5, a6);
      v27 = v30;
      if ( !v26 )
      {
        if ( n )
        {
          sub_16BD1C0("Allocation failed");
          v27 = v30;
          v26 = 0;
        }
        else
        {
          v26 = sub_13A3880(1u);
          v27 = v30;
        }
      }
      *(_QWORD *)(v7 + 8) = v27;
      *(_QWORD *)v7 = v26;
      sub_13A4C60(v7, 0);
      v28 = *(_QWORD *)(v7 + 8) - (unsigned int)v23;
      if ( v28 )
        memset((void *)(*(_QWORD *)v7 + 8LL * (unsigned int)v23), -v24, 8 * v28);
    }
    result = *(unsigned int *)(v7 + 16);
    if ( a2 > (unsigned int)result )
    {
      sub_13A4C60(v7, v24);
      result = *(unsigned int *)(v7 + 16);
    }
    *(_DWORD *)(v7 + 16) = a2;
    if ( a2 < (unsigned int)result || a3 )
      return sub_13A4C60(v7, 0);
  }
  return result;
}
