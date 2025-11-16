// Function: sub_C93CF0
// Address: 0xc93cf0
//
__int64 __fastcall sub_C93CF0(__m128i *a1, unsigned int a2, unsigned __int64 *a3)
{
  unsigned int v4; // ebx
  __int64 result; // rax
  unsigned __int64 v6; // rdx
  __m128i v7; // xmm3
  int v8; // r15d
  unsigned int v9; // r12d
  int v10; // r13d
  unsigned int v11; // r15d
  unsigned int v12; // eax
  __int64 v13; // rdx
  char *v14; // rsi
  char v15; // al
  unsigned int v16; // eax
  __int64 v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // rax
  bool v20; // cc
  unsigned int v21; // edi
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rcx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned __int8 v29; // [rsp+10h] [rbp-80h]
  unsigned __int8 v30; // [rsp+10h] [rbp-80h]
  unsigned int v31; // [rsp+10h] [rbp-80h]
  unsigned __int128 v32; // [rsp+20h] [rbp-70h] BYREF
  __int64 v33; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-58h]
  _QWORD *v35; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-48h]
  _QWORD *v37; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-38h]

  v4 = a2;
  v32 = (unsigned __int128)_mm_loadu_si128(a1);
  if ( !a2 )
    v4 = sub_C92F70((__int64 *)&v32);
  result = 1;
  if ( !*((_QWORD *)&v32 + 1) )
    return result;
  v6 = sub_C93580(&v32, 48, 0);
  if ( v6 >= *((_QWORD *)&v32 + 1) )
  {
    v32 = (unsigned __int64)(*((_QWORD *)&v32 + 1) + v32);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
    {
      if ( *a3 )
        j_j___libc_free_0_0(*a3);
    }
    v7 = _mm_loadu_si128((const __m128i *)&v32);
    *a3 = 0;
    *((_DWORD *)a3 + 2) = 64;
    *a1 = v7;
    return 0;
  }
  *(_QWORD *)&v32 = v32 + v6;
  if ( *((_QWORD *)&v32 + 1) - v6 == -1 )
  {
    v8 = -1;
    *((_QWORD *)&v32 + 1) = -1;
  }
  else
  {
    v8 = DWORD2(v32) - v6;
    *((_QWORD *)&v32 + 1) -= v6;
  }
  if ( v4 <= 1 )
  {
    v12 = *((_DWORD *)a3 + 2);
    if ( !v12 )
    {
      v11 = 0;
      v9 = 0;
      v10 = 1;
LABEL_17:
      v12 = v11;
      goto LABEL_18;
    }
    v9 = 0;
    v10 = 1;
  }
  else
  {
    v9 = 0;
    do
      v10 = 1 << ++v9;
    while ( 1 << v9 < v4 );
    v11 = v9 * v8;
    v12 = *((_DWORD *)a3 + 2);
    if ( v12 <= v11 )
    {
      if ( v12 < v11 )
      {
        sub_C449B0((__int64)&v37, (const void **)a3, v11);
        if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
          j_j___libc_free_0_0(*a3);
        *a3 = (unsigned __int64)v37;
        *((_DWORD *)a3 + 2) = v38;
      }
      goto LABEL_17;
    }
  }
LABEL_18:
  v34 = 1;
  v33 = 0;
  v36 = 1;
  v35 = 0;
  if ( v4 == v10 )
  {
    if ( *((_DWORD *)a3 + 2) > 0x40u )
    {
LABEL_20:
      *(_QWORD *)*a3 = 0;
      memset((void *)(*a3 + 8), 0, 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
      goto LABEL_21;
    }
  }
  else
  {
    v38 = v12;
    if ( v12 > 0x40 )
    {
      v31 = v12;
      sub_C43690((__int64)&v37, v4, 0);
      if ( v34 > 0x40 && v33 )
      {
        j_j___libc_free_0_0(v33);
        v33 = (__int64)v37;
        v27 = v38;
        v38 = v31;
        v34 = v27;
      }
      else
      {
        v33 = (__int64)v37;
        v26 = v38;
        v38 = v31;
        v34 = v26;
      }
      sub_C43690((__int64)&v37, 0, 0);
      if ( v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
    }
    else
    {
      v33 = v4;
      v34 = v12;
      v37 = 0;
    }
    v20 = *((_DWORD *)a3 + 2) <= 0x40u;
    v35 = v37;
    v36 = v38;
    if ( !v20 )
      goto LABEL_20;
  }
  *a3 = 0;
LABEL_21:
  v13 = *((_QWORD *)&v32 + 1);
  v14 = (char *)v32;
  if ( *((_QWORD *)&v32 + 1) )
  {
    do
    {
      v15 = *v14;
      if ( *v14 <= 47 )
        goto LABEL_23;
      if ( v15 <= 57 )
      {
        v16 = (char)(v15 - 48);
        goto LABEL_25;
      }
      if ( v15 <= 96 )
      {
LABEL_23:
        if ( (unsigned __int8)(v15 - 65) > 0x19u )
          goto LABEL_35;
        v16 = (char)(v15 - 55);
      }
      else
      {
        if ( v15 > 122 )
          goto LABEL_35;
        v16 = (char)(v15 - 87);
      }
LABEL_25:
      if ( v4 <= v16 )
        goto LABEL_35;
      v17 = v16;
      if ( v4 == v10 )
      {
        v21 = *((_DWORD *)a3 + 2);
        if ( v21 > 0x40 )
        {
          sub_C47690((__int64 *)a3, v9);
          v21 = *((_DWORD *)a3 + 2);
          if ( v21 > 0x40 )
          {
            *(_QWORD *)*a3 |= v17;
            v14 = (char *)v32;
            goto LABEL_32;
          }
          v14 = (char *)v32;
          v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
        }
        else
        {
          v22 = 0;
          if ( v9 != v21 )
            v22 = *a3 << v9;
          v23 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v21) & v22;
          v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
          if ( !v21 )
            v23 = 0;
          *a3 = v23;
        }
        v25 = (v17 | *a3) & v24;
        if ( !v21 )
          v25 = 0;
        *a3 = v25;
      }
      else
      {
        sub_C47360((__int64)a3, &v33);
        if ( v36 > 0x40 )
        {
          *v35 = v17;
          memset(v35 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v36 + 63) >> 6) - 8);
        }
        else
        {
          v18 = (_QWORD *)(v17 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v36));
          if ( !v36 )
            v18 = 0;
          v35 = v18;
        }
        sub_C45EE0((__int64)a3, (__int64 *)&v35);
        v14 = (char *)v32;
      }
LABEL_32:
      v19 = *((_QWORD *)&v32 + 1);
      if ( !*((_QWORD *)&v32 + 1) )
      {
        *(_QWORD *)&v32 = v14;
        break;
      }
      v13 = *((_QWORD *)&v32 + 1) - 1LL;
      v32 = __PAIR128__(*((unsigned __int64 *)&v32 + 1), (unsigned __int64)++v14) + __PAIR128__(-1, 0);
    }
    while ( v19 != 1 );
  }
  v13 = 0;
LABEL_35:
  result = 1;
  if ( a1->m128i_i64[1] != v13 )
  {
    result = 0;
    *a1 = _mm_loadu_si128((const __m128i *)&v32);
  }
  if ( v36 > 0x40 && v35 )
  {
    v29 = result;
    j_j___libc_free_0_0(v35);
    result = v29;
  }
  if ( v34 > 0x40 && v33 )
  {
    v30 = result;
    j_j___libc_free_0_0(v33);
    return v30;
  }
  return result;
}
