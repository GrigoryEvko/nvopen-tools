// Function: sub_617930
// Address: 0x617930
//
__int64 __fastcall sub_617930(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r15
  const void *v3; // rbx
  size_t v4; // r14
  __int64 v5; // rax
  char v6; // si
  size_t v7; // r12
  const void *v8; // r13
  size_t v9; // rdx
  signed __int64 v10; // rax
  __int64 v11; // r8
  int v12; // eax
  unsigned int v14; // r14d
  __int64 v15; // rdx
  __int64 v16; // rax
  size_t v17; // r15
  size_t v18; // r14
  size_t v19; // rdx
  __int64 v20; // rax
  size_t v21; // rax
  _QWORD *v22; // [rsp+10h] [rbp-50h]
  size_t na; // [rsp+20h] [rbp-40h]
  __int64 nb; // [rsp+20h] [rbp-40h]
  __int64 n; // [rsp+20h] [rbp-40h]
  __int64 nc; // [rsp+20h] [rbp-40h]

  v2 = a1[2];
  v22 = a1 + 1;
  if ( !v2 )
  {
    v2 = (__int64)(a1 + 1);
    goto LABEL_27;
  }
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v7 = *(_QWORD *)(v2 + 40);
    v8 = *(const void **)(v2 + 32);
    v9 = v7;
    if ( v4 <= v7 )
      v9 = v4;
    if ( v9 )
    {
      na = v9;
      LODWORD(v10) = memcmp(v3, *(const void **)(v2 + 32), v9);
      v9 = na;
      if ( (_DWORD)v10 )
        goto LABEL_11;
    }
    v10 = v4 - v7;
    if ( (__int64)(v4 - v7) >= 0x80000000LL )
      break;
    if ( v10 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_11:
      if ( (int)v10 >= 0 )
        break;
    }
    v5 = *(_QWORD *)(v2 + 16);
    v6 = 1;
    if ( !v5 )
      goto LABEL_13;
LABEL_4:
    v2 = v5;
  }
  v5 = *(_QWORD *)(v2 + 24);
  v6 = 0;
  if ( v5 )
    goto LABEL_4;
LABEL_13:
  v11 = v2;
  if ( !v6 )
    goto LABEL_14;
LABEL_27:
  if ( v2 == a1[3] )
  {
    v11 = v2;
    v14 = 1;
    if ( v22 != (_QWORD *)v2 )
      goto LABEL_33;
    goto LABEL_23;
  }
  v16 = sub_220EF80(v2);
  v11 = v2;
  v7 = *(_QWORD *)(v16 + 40);
  v8 = *(const void **)(v16 + 32);
  v2 = v16;
  v4 = *(_QWORD *)(a2 + 8);
  v3 = *(const void **)a2;
  v9 = v7;
  if ( v4 <= v7 )
    v9 = *(_QWORD *)(a2 + 8);
LABEL_14:
  if ( v9 && (nb = v11, v12 = memcmp(v8, v3, v9), v11 = nb, v12) )
  {
LABEL_19:
    if ( v12 < 0 )
      goto LABEL_21;
  }
  else if ( (__int64)(v7 - v4) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v7 - v4) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v12 = v7 - v4;
      goto LABEL_19;
    }
LABEL_21:
    if ( v11 )
    {
      v14 = 1;
      if ( v22 == (_QWORD *)v11 )
        goto LABEL_23;
LABEL_33:
      v17 = *(_QWORD *)(v11 + 40);
      v18 = *(_QWORD *)(a2 + 8);
      v19 = v17;
      if ( v18 <= v17 )
        v19 = *(_QWORD *)(a2 + 8);
      if ( v19
        && (nc = v11, LODWORD(v20) = memcmp(*(const void **)a2, *(const void **)(v11 + 32), v19), v11 = nc, (_DWORD)v20) )
      {
LABEL_39:
        v14 = (unsigned int)v20 >> 31;
      }
      else
      {
        v21 = v18;
        v14 = 0;
        v20 = v21 - v17;
        if ( v20 <= 0x7FFFFFFF )
        {
          if ( v20 >= (__int64)0xFFFFFFFF80000000LL )
            goto LABEL_39;
          v14 = 1;
        }
      }
LABEL_23:
      n = v11;
      v2 = sub_22077B0(64);
      *(_QWORD *)(v2 + 32) = v2 + 48;
      if ( *(_QWORD *)a2 == a2 + 16 )
      {
        *(__m128i *)(v2 + 48) = _mm_loadu_si128((const __m128i *)(a2 + 16));
      }
      else
      {
        *(_QWORD *)(v2 + 32) = *(_QWORD *)a2;
        *(_QWORD *)(v2 + 48) = *(_QWORD *)(a2 + 16);
      }
      v15 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)a2 = a2 + 16;
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(v2 + 40) = v15;
      *(_BYTE *)(a2 + 16) = 0;
      sub_220F040(v14, v2, n, v22);
      ++a1[5];
    }
    else
    {
      return 0;
    }
  }
  return v2;
}
