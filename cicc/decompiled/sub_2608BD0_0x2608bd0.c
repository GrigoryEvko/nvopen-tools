// Function: sub_2608BD0
// Address: 0x2608bd0
//
int __fastcall sub_2608BD0(__int64 **a1, _QWORD **a2)
{
  __int64 **v2; // r13
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v6; // rax
  unsigned int v7; // ebx
  _QWORD *v8; // rdi
  unsigned __int64 v9; // rax
  int v10; // esi
  __m128i *v11; // r9
  __int64 *v12; // r13
  unsigned int v13; // ebx
  void *v14; // r10
  const void *v15; // rdi
  int v16; // r11d
  unsigned int v17; // r8d
  size_t v18; // rdx
  __int64 *v19; // r15
  const void *v20; // rcx
  unsigned int v21; // r8d
  __int64 *v22; // rsi
  int v23; // eax
  size_t v25; // [rsp+0h] [rbp-A0h]
  const void *v26; // [rsp+8h] [rbp-98h]
  int v28; // [rsp+20h] [rbp-80h]
  unsigned int v29; // [rsp+24h] [rbp-7Ch]
  void *v30; // [rsp+28h] [rbp-78h]
  __m128i *v31; // [rsp+30h] [rbp-70h]
  __int64 v32; // [rsp+40h] [rbp-60h]
  __int64 *v33; // [rsp+48h] [rbp-58h]
  __m128i *v34; // [rsp+58h] [rbp-48h] BYREF
  void *s1[2]; // [rsp+60h] [rbp-40h] BYREF

  v2 = a1;
  v3 = *a1;
  v32 = (__int64)(a1 + 17);
  v4 = (__int64)&v34;
  if ( *a1 != a1[1] )
  {
    v33 = a1[1];
    while ( 1 )
    {
      v6 = *v3;
      v7 = *((_DWORD *)a1 + 40);
      v8 = *(_QWORD **)(*v3 + 200);
      s1[0] = v8;
      v9 = *(unsigned int *)(v6 + 208);
      s1[1] = (void *)v9;
      if ( !v7 )
      {
        a1[17] = (__int64 *)((char *)a1[17] + 1);
        v34 = 0;
LABEL_5:
        v10 = 2 * v7;
        goto LABEL_6;
      }
      v12 = a1[18];
      v13 = v7 - 1;
      LODWORD(v4) = sub_939680(v8, (__int64)v8 + 4 * v9);
      v14 = s1[1];
      v15 = s1[0];
      v11 = 0;
      v16 = 1;
      v17 = v13 & v4;
      v18 = 4 * (__int64)s1[1];
      while ( 1 )
      {
        v19 = &v12[2 * v17];
        v20 = (const void *)*v19;
        if ( *v19 == -1 )
          break;
        if ( v20 == (const void *)-2LL )
        {
          if ( v15 == (const void *)-2LL )
            goto LABEL_19;
        }
        else
        {
          if ( v14 != (void *)v19[1] )
            goto LABEL_17;
          v28 = v16;
          v29 = v17;
          v30 = v14;
          v31 = v11;
          if ( !v18 )
            goto LABEL_19;
          v25 = v18;
          v26 = (const void *)*v19;
          LODWORD(v4) = memcmp(v15, v20, v18);
          v20 = v26;
          v18 = v25;
          v11 = v31;
          v14 = v30;
          v17 = v29;
          v16 = v28;
          if ( !(_DWORD)v4 )
            goto LABEL_19;
        }
        if ( v20 == (const void *)-2LL && !v11 )
          v11 = (__m128i *)v19;
LABEL_17:
        v21 = v16 + v17;
        ++v16;
        v17 = v13 & v21;
      }
      if ( v15 == (const void *)-1LL )
      {
LABEL_19:
        if ( v33 != ++v3 )
          continue;
        goto LABEL_10;
      }
      v23 = *((_DWORD *)a1 + 38);
      v7 = *((_DWORD *)a1 + 40);
      if ( !v11 )
        v11 = (__m128i *)&v12[2 * v17];
      a1[17] = (__int64 *)((char *)a1[17] + 1);
      LODWORD(v4) = v23 + 1;
      v34 = v11;
      if ( 4 * (int)v4 >= 3 * v7 )
        goto LABEL_5;
      if ( v7 - ((_DWORD)v4 + *((_DWORD *)a1 + 39)) > v7 >> 3 )
        goto LABEL_7;
      v10 = v7;
LABEL_6:
      sub_2608A40(v32, v10);
      sub_2608900(v32, (__int64)s1, &v34);
      v11 = v34;
      LODWORD(v4) = *((_DWORD *)a1 + 38) + 1;
LABEL_7:
      *((_DWORD *)a1 + 38) = v4;
      if ( v11->m128i_i64[0] != -1 )
        --*((_DWORD *)a1 + 39);
      ++v3;
      *v11 = _mm_loadu_si128((const __m128i *)s1);
      if ( v33 == v3 )
      {
LABEL_10:
        v2 = a1;
        break;
      }
    }
  }
  if ( *((_DWORD *)v2 + 38) > 1u )
  {
    v4 = sub_BCB2D0(*a2);
    v22 = v2[4];
    s1[0] = (void *)v4;
    if ( v22 == v2[5] )
    {
      LODWORD(v4) = (unsigned int)sub_9183A0((__int64)(v2 + 3), v22, s1);
    }
    else
    {
      if ( v22 )
      {
        *v22 = v4;
        v22 = v2[4];
      }
      v2[4] = v22 + 1;
    }
  }
  return v4;
}
