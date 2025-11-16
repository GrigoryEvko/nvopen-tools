// Function: sub_228BF90
// Address: 0x228bf90
//
unsigned __int64 __fastcall sub_228BF90(
        unsigned __int64 *a1,
        unsigned int a2,
        unsigned __int8 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v8; // rbx
  bool v9; // zf
  __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  unsigned __int64 result; // rax
  _QWORD *v14; // rax
  __int64 v15; // r9
  _QWORD *v16; // rbx
  __int64 *v17; // rsi
  unsigned int v18; // r15d
  __int64 v19; // r8
  __int64 *v20; // rax
  int v21; // r13d
  unsigned __int64 v22; // r9
  __int64 v23; // rcx
  unsigned __int64 v24; // r8
  int v25; // ecx
  unsigned __int64 v26; // rdx
  int v27; // r13d
  __int64 v28; // rax
  __int64 *v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rcx
  __int64 v35; // r14
  int v36; // r13d
  _QWORD *v37; // rax
  __int64 v38; // rdx

  v8 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    if ( a2 > 0x39 )
    {
      v14 = (_QWORD *)sub_22077B0(0x48u);
      v16 = v14;
      if ( v14 )
      {
        v17 = v14 + 2;
        *v14 = v14 + 2;
        v18 = (a2 + 63) >> 6;
        v19 = -(__int64)a3;
        v14[1] = 0x600000000LL;
        if ( v18 > 6 )
        {
          sub_C8D5F0((__int64)v14, v17, v18, 8u, v19, v15);
          v37 = (_QWORD *)*v16;
          v38 = *v16 + 8LL * v18;
          do
            *v37++ = -(__int64)a3;
          while ( (_QWORD *)v38 != v37 );
        }
        else if ( v18 )
        {
          v20 = &v17[v18];
          do
            *v17++ = v19;
          while ( v20 != v17 );
        }
        *((_DWORD *)v16 + 2) = v18;
        *((_DWORD *)v16 + 16) = a2;
        if ( a3 )
        {
          v21 = a2 & 0x3F;
          if ( v21 )
            *(_QWORD *)(*v16 + 8LL * *((unsigned int *)v16 + 2) - 8) &= ~(-1LL << v21);
        }
      }
      v22 = *a1 >> 58;
      result = -1LL << v22;
      v23 = 0;
      v24 = (*a1 >> 1) & ~(-1LL << v22);
      if ( v22 )
      {
        do
        {
          result = *(_QWORD *)*v16 & ~(1LL << v23);
          if ( ((v24 >> v23) & 1) != 0 )
            result = (1LL << v23) | *(_QWORD *)*v16;
          ++v23;
          *(_QWORD *)*v16 = result;
        }
        while ( v23 != v22 );
      }
      *a1 = (unsigned __int64)v16;
    }
    else
    {
      v9 = a3 == 0;
      v10 = 0;
      if ( !v9 )
        v10 = -1LL << (v8 >> 58);
      v11 = ((unsigned __int64)a2 << 57) | ~(-1LL << (v8 >> 58)) & (v8 >> 1);
      v12 = (2 * v11) >> 58;
      result = ~(-1LL << v12);
      *a1 = 2 * ((v12 << 57) | result & (v10 | result & v11)) + 1;
    }
    return result;
  }
  v25 = *(_DWORD *)(v8 + 64) & 0x3F;
  if ( v25 )
  {
    v28 = -1LL << v25;
    v29 = (__int64 *)(*(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8) - 8);
    v30 = ~v28;
    v31 = *v29 | v28;
    v32 = *v29 & v30;
    if ( !a3 )
      v31 = v32;
    *v29 = v31;
  }
  v26 = *(unsigned int *)(v8 + 8);
  *(_DWORD *)(v8 + 64) = a2;
  result = (a2 + 63) >> 6;
  if ( result == v26 )
    goto LABEL_24;
  if ( result < v26 )
  {
    *(_DWORD *)(v8 + 8) = result;
LABEL_24:
    LOBYTE(v27) = a2 & 0x3F;
    if ( (a2 & 0x3F) == 0 )
      return result;
LABEL_25:
    result = ~(-1LL << v27);
    *(_QWORD *)(*(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8) - 8) &= result;
    return result;
  }
  v33 = result - v26;
  if ( result > *(unsigned int *)(v8 + 12) )
  {
    sub_C8D5F0(v8, (const void *)(v8 + 16), (a2 + 63) >> 6, 8u, result, a6);
    v26 = *(unsigned int *)(v8 + 8);
  }
  result = *(_QWORD *)v8 + 8 * v26;
  v34 = result + 8 * v33;
  if ( result != v34 )
  {
    v35 = -(__int64)a3;
    do
    {
      *(_QWORD *)result = v35;
      result += 8LL;
    }
    while ( v34 != result );
    LODWORD(v26) = *(_DWORD *)(v8 + 8);
  }
  v36 = *(_DWORD *)(v8 + 64);
  *(_DWORD *)(v8 + 8) = v33 + v26;
  v27 = v36 & 0x3F;
  if ( v27 )
    goto LABEL_25;
  return result;
}
