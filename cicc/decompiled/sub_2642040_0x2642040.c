// Function: sub_2642040
// Address: 0x2642040
//
unsigned __int64 __fastcall sub_2642040(_QWORD *a1, __int64 a2, const __m128i **a3)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // rsi
  __m128i v10; // xmm0
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  bool v15; // al
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  char v21; // di
  unsigned __int64 v22; // rax
  unsigned int v23; // edi
  unsigned int v24; // eax
  unsigned __int64 v25; // [rsp+18h] [rbp-38h]
  _QWORD *v26; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(0x60u);
  v7 = a1 + 1;
  v8 = v6;
  v9 = (unsigned __int64 *)(v6 + 32);
  v10 = _mm_loadu_si128(*a3);
  v6 += 56;
  *(_DWORD *)(v8 + 56) = 0;
  *(_QWORD *)(v8 + 64) = 0;
  *(_QWORD *)(v8 + 72) = v6;
  *(_QWORD *)(v8 + 80) = v6;
  *(_QWORD *)(v8 + 88) = 0;
  *(__m128i *)(v8 + 32) = v10;
  if ( a1 + 1 != (_QWORD *)a2 )
  {
    v11 = *(_QWORD *)(v8 + 32);
    v12 = *(_QWORD *)(a2 + 32);
    v13 = a2;
    if ( v11 < v12 )
    {
LABEL_3:
      if ( a1[3] != a2 )
      {
        v14 = sub_220EF80(a2);
        v7 = a1 + 1;
        if ( v11 <= *(_QWORD *)(v14 + 32)
          && (v11 != *(_QWORD *)(v14 + 32) || *(_DWORD *)(v14 + 40) >= *(_DWORD *)(v8 + 40)) )
        {
LABEL_13:
          v26 = v7;
          v18 = sub_263E300((__int64)a1, v9);
          v7 = v26;
          a2 = v18;
          v13 = v19;
          if ( !v19 )
          {
LABEL_14:
            sub_2641CE0(0);
            j_j___libc_free_0(v8);
            return a2;
          }
          goto LABEL_16;
        }
        if ( !*(_QWORD *)(v14 + 24) )
        {
          v13 = v14;
          goto LABEL_7;
        }
      }
LABEL_16:
      v15 = a2 != 0;
      goto LABEL_17;
    }
    if ( v11 == v12 )
    {
      v23 = *(_DWORD *)(v8 + 40);
      v24 = *(_DWORD *)(a2 + 40);
      if ( v23 < v24 )
        goto LABEL_3;
    }
    else
    {
      if ( v11 > v12 )
      {
LABEL_10:
        v25 = *(_QWORD *)(a2 + 32);
        if ( a1[4] != a2 )
        {
          v16 = sub_220EEE0(a2);
          v7 = a1 + 1;
          v17 = v25;
          if ( v11 < *(_QWORD *)(v16 + 32)
            || v11 == *(_QWORD *)(v16 + 32) && *(_DWORD *)(v8 + 40) < *(_DWORD *)(v16 + 40) )
          {
            if ( *(_QWORD *)(a2 + 24) )
            {
              v13 = v16;
              v21 = 1;
              goto LABEL_20;
            }
            goto LABEL_33;
          }
          goto LABEL_13;
        }
        a2 = 0;
        goto LABEL_16;
      }
      v23 = *(_DWORD *)(v8 + 40);
      v24 = *(_DWORD *)(a2 + 40);
    }
    if ( v23 <= v24 )
      goto LABEL_14;
    goto LABEL_10;
  }
  if ( !a1[5] )
    goto LABEL_13;
  v13 = a1[4];
  v22 = *(_QWORD *)(v8 + 32);
  if ( *(_QWORD *)(v13 + 32) >= v22 )
  {
    if ( *(_QWORD *)(v13 + 32) == v22 && *(_DWORD *)(v13 + 40) < *(_DWORD *)(v8 + 40) )
    {
      v15 = 0;
      goto LABEL_17;
    }
    goto LABEL_13;
  }
LABEL_7:
  v15 = 0;
LABEL_17:
  if ( v7 != (_QWORD *)v13 && !v15 )
  {
    v11 = *(_QWORD *)(v8 + 32);
    v17 = *(_QWORD *)(v13 + 32);
    v21 = 1;
    if ( v17 > v11 )
      goto LABEL_20;
LABEL_33:
    v21 = 0;
    if ( v17 == v11 )
      v21 = *(_DWORD *)(v8 + 40) < *(_DWORD *)(v13 + 40);
    goto LABEL_20;
  }
  v21 = 1;
LABEL_20:
  sub_220F040(v21, v8, (_QWORD *)v13, v7);
  ++a1[5];
  return v8;
}
