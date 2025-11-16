// Function: sub_34202D0
// Address: 0x34202d0
//
unsigned __int8 *__fastcall sub_34202D0(_DWORD *a1, __m128i *a2)
{
  bool v3; // zf
  unsigned __int8 *v4; // r15
  __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // r9
  __int64 *v9; // r15
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  __int64 *v15; // rdx
  _QWORD *v16; // rax
  unsigned int i; // eax
  __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // r15
  __int64 *v22; // r14
  __int64 v23; // rax
  __int64 *v24; // r15
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 *v27; // r15
  unsigned __int64 v28; // r14
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // rsi
  __int128 v32; // [rsp-10h] [rbp-1C0h]
  __int64 v33; // [rsp+0h] [rbp-1B0h]
  __int64 v34; // [rsp+0h] [rbp-1B0h]
  __int64 v35; // [rsp+8h] [rbp-1A8h]
  __int64 *v36; // [rsp+18h] [rbp-198h]
  __m128i v37; // [rsp+20h] [rbp-190h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-180h] BYREF
  __int64 (__fastcall *v39)(unsigned __int64 *, const __m128i **, int); // [rsp+40h] [rbp-170h]
  __m128i *(__fastcall *v40)(__int64 **, const __m128i *, __int64 *, __int64, __int64, __int64); // [rsp+48h] [rbp-168h]
  __int64 *v41; // [rsp+50h] [rbp-160h] BYREF
  __int64 v42; // [rsp+58h] [rbp-158h]
  _BYTE v43[48]; // [rsp+60h] [rbp-150h] BYREF
  __int64 *v44; // [rsp+90h] [rbp-120h] BYREF
  __int64 v45; // [rsp+98h] [rbp-118h]
  _BYTE v46[64]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v47; // [rsp+E0h] [rbp-D0h] BYREF
  void *s; // [rsp+E8h] [rbp-C8h]
  _BYTE v49[12]; // [rsp+F0h] [rbp-C0h]
  char v50; // [rsp+FCh] [rbp-B4h]
  char v51; // [rsp+100h] [rbp-B0h] BYREF

  s = &v51;
  v3 = a1[2] == 1;
  v45 = 0x800000000LL;
  v41 = (__int64 *)v43;
  v36 = (__int64 *)a2;
  v47 = 0;
  *(_QWORD *)v49 = 16;
  *(_DWORD *)&v49[8] = 0;
  v50 = 1;
  v44 = (__int64 *)v46;
  v42 = 0x300000000LL;
  if ( v3 )
  {
    v4 = **(unsigned __int8 ***)(**(_QWORD **)a1 + 40LL);
    goto LABEL_3;
  }
  v6 = 24;
  v39 = 0;
  v7 = (_QWORD *)sub_22077B0(0x18u);
  if ( v7 )
  {
    *v7 = &v47;
    v7[1] = v38;
    v7[2] = &v41;
  }
  v38[0] = v7;
  v8 = *(_QWORD *)a1;
  v40 = sub_341F730;
  v39 = sub_341EB50;
  v9 = (__int64 *)v8;
  v10 = (__int64 *)(v8 + 8LL * (unsigned int)a1[2]);
  if ( v10 != (__int64 *)v8 )
  {
    while ( 1 )
    {
      v11 = (unsigned int)v45;
      v12 = HIDWORD(v45);
      v13 = *v9;
      v14 = (unsigned int)v45 + 1LL;
      if ( v14 > HIDWORD(v45) )
      {
        v6 = (__int64)&v44;
        a2 = (__m128i *)v46;
        v33 = *v9;
        sub_C8D5F0((__int64)&v44, v46, v14, 8u, v13, v8);
        v11 = (unsigned int)v45;
        v13 = v33;
      }
      v15 = v44;
      v44[v11] = v13;
      LODWORD(v45) = v45 + 1;
      if ( !v50 )
        goto LABEL_27;
      v16 = s;
      v12 = *(unsigned int *)&v49[4];
      v15 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v49[4]);
      if ( s == v15 )
      {
LABEL_29:
        if ( *(_DWORD *)&v49[4] >= *(_DWORD *)v49 )
        {
LABEL_27:
          a2 = (__m128i *)v13;
          v6 = (__int64)&v47;
          ++v9;
          sub_C8CC70((__int64)&v47, v13, (__int64)v15, v12, v13, v8);
          if ( v10 == v9 )
            break;
        }
        else
        {
          ++v9;
          ++*(_DWORD *)&v49[4];
          *v15 = v13;
          ++v47;
          if ( v10 == v9 )
            break;
        }
      }
      else
      {
        while ( v13 != *v16 )
        {
          if ( v15 == ++v16 )
            goto LABEL_29;
        }
        if ( v10 == ++v9 )
          break;
      }
    }
  }
  for ( i = v45; (_DWORD)v45; i = v45 )
  {
    v18 = v44[i - 1];
    LODWORD(v45) = i - 1;
    v37 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v18 + 40));
    if ( !v39 )
      sub_4263D6(v6, a2, v18);
    v6 = (__int64)v38;
    a2 = &v37;
    ((void (__fastcall *)(_QWORD *, __m128i *))v40)(v38, &v37);
  }
  v19 = v42;
  v4 = (unsigned __int8 *)(v36 + 36);
  if ( (_DWORD)v42 )
  {
    ++v47;
    if ( !v50 )
    {
      v20 = 4 * (*(_DWORD *)&v49[4] - *(_DWORD *)&v49[8]);
      if ( v20 < 0x20 )
        v20 = 32;
      if ( *(_DWORD *)v49 > v20 )
      {
        sub_C8C990((__int64)&v47, (__int64)v36);
        v19 = v42;
LABEL_38:
        v21 = v19;
        v22 = v41;
        v23 = (unsigned int)v45;
        v24 = &v41[2 * v21];
        if ( v24 != v41 )
        {
          do
          {
            v25 = *v22;
            if ( v23 + 1 > (unsigned __int64)HIDWORD(v45) )
            {
              v34 = *v22;
              sub_C8D5F0((__int64)&v44, v46, v23 + 1, 8u, v25, v8);
              v23 = (unsigned int)v45;
              v25 = v34;
            }
            v22 += 2;
            v44[v23] = v25;
            v23 = (unsigned int)(v45 + 1);
            LODWORD(v45) = v45 + 1;
          }
          while ( v24 != v22 );
        }
        v26 = *(_QWORD *)a1;
        v27 = *(__int64 **)a1;
        v35 = *(_QWORD *)a1 + 8LL * (unsigned int)a1[2];
        if ( v35 == *(_QWORD *)a1 )
        {
LABEL_47:
          if ( (unsigned int)v42 == 1 )
          {
            v4 = (unsigned __int8 *)*v41;
          }
          else
          {
            v28 = (unsigned __int64)v41;
            v29 = (unsigned int)v42;
            v30 = **(_QWORD **)a1;
            v31 = *(_QWORD *)(v30 + 80);
            v37.m128i_i64[0] = v31;
            if ( v31 )
              sub_B96E90((__int64)&v37, v31, 1);
            *((_QWORD *)&v32 + 1) = v29;
            *(_QWORD *)&v32 = v28;
            v37.m128i_i32[2] = *(_DWORD *)(v30 + 72);
            v4 = sub_33FC220(v36, 2, (__int64)&v37, 1, 0, v26, v32);
            if ( v37.m128i_i64[0] )
              sub_B91220((__int64)&v37, v37.m128i_i64[0]);
          }
        }
        else
        {
          while ( !(unsigned __int8)sub_3285B00(*v27, (__int64)&v47, (__int64)&v44, 0x2000u, 1, v26) )
          {
            if ( (__int64 *)v35 == ++v27 )
              goto LABEL_47;
          }
          v4 = 0;
        }
        goto LABEL_20;
      }
      memset(s, -1, 8LL * *(unsigned int *)v49);
      v19 = v42;
    }
    *(_QWORD *)&v49[4] = 0;
    goto LABEL_38;
  }
LABEL_20:
  if ( v39 )
    v39(v38, (const __m128i **)v38, 3);
  if ( v41 != (__int64 *)v43 )
    _libc_free((unsigned __int64)v41);
  if ( v44 != (__int64 *)v46 )
  {
    _libc_free((unsigned __int64)v44);
    if ( v50 )
      return v4;
LABEL_26:
    _libc_free((unsigned __int64)s);
    return v4;
  }
LABEL_3:
  if ( !v50 )
    goto LABEL_26;
  return v4;
}
