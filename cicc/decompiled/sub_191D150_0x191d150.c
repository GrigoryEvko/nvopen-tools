// Function: sub_191D150
// Address: 0x191d150
//
__int64 __fastcall sub_191D150(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v15; // rdi
  double v16; // xmm4_8
  double v17; // xmm5_8
  _QWORD *v18; // r9
  __int64 v19; // r15
  unsigned int v20; // r14d
  double v21; // xmm4_8
  double v22; // xmm5_8
  _BYTE *v23; // rdi
  _QWORD **i; // r14
  __int64 *v26; // r14
  double v27; // xmm4_8
  double v28; // xmm5_8
  unsigned __int8 v29; // al
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  char v34; // al
  int v35; // r9d
  __int64 v36; // rax
  __int64 *v37; // rdi
  _QWORD *v38; // [rsp+18h] [rbp-E78h]
  __int64 *v39; // [rsp+20h] [rbp-E70h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-E68h] BYREF
  _BYTE *v41; // [rsp+30h] [rbp-E60h] BYREF
  __int64 v42; // [rsp+38h] [rbp-E58h]
  _BYTE v43[512]; // [rsp+40h] [rbp-E50h] BYREF
  _BYTE *v44; // [rsp+240h] [rbp-C50h] BYREF
  __int64 v45; // [rsp+248h] [rbp-C48h]
  _BYTE v46[1536]; // [rsp+250h] [rbp-C40h] BYREF
  _BYTE *v47; // [rsp+850h] [rbp-640h] BYREF
  __int64 v48; // [rsp+858h] [rbp-638h]
  _BYTE v49[1584]; // [rsp+860h] [rbp-630h] BYREF

  v15 = *(_QWORD *)a1;
  v45 = 0x4000000000LL;
  v44 = v46;
  sub_141E480(v15, (__int64)a2, (__int64)&v44, 1, a13);
  v18 = &v44;
  if ( (unsigned int)v45 > 0x32 )
  {
    v23 = v44;
    v20 = 0;
    goto LABEL_14;
  }
  if ( (_DWORD)v45 == 1 )
  {
    v23 = v44;
    v20 = 0;
    if ( (*((_DWORD *)v44 + 2) & 7u) - 1 > 1 )
      goto LABEL_14;
    v19 = *(a2 - 3);
    if ( *(_BYTE *)(v19 + 16) != 56 )
      goto LABEL_4;
    goto LABEL_21;
  }
  v19 = *(a2 - 3);
  if ( *(_BYTE *)(v19 + 16) == 56 )
  {
LABEL_21:
    for ( i = (_QWORD **)(v19 + 24 * (1LL - (*(_DWORD *)(v19 + 20) & 0xFFFFFFF))); (_QWORD **)v19 != i; i += 3 )
    {
      if ( *((_BYTE *)*i + 16) > 0x17u )
      {
        v38 = v18;
        sub_191B610(a1, *i, 0, a3, a4, a5, a6, v16, v17, a9, a10);
        v18 = v38;
      }
    }
  }
LABEL_4:
  v20 = 0;
  v47 = v49;
  v48 = 0x4000000000LL;
  v41 = v43;
  v42 = 0x4000000000LL;
  sub_190CA00(a1, a2, (__int64)v18, (__int64)&v47, (__int64)&v41, (int)v18);
  if ( (_DWORD)v48 )
  {
    if ( (_DWORD)v42 )
    {
      v20 = (unsigned __int8)byte_4FAEF80;
      if ( byte_4FAEF80 )
      {
        v20 = (unsigned __int8)byte_4FAEEA0;
        if ( byte_4FAEEA0 )
          v20 = sub_19152E0(a1, (__int64)a2, (__int64)&v47, (__int64)&v41, a3, a4, a5, a6, v21, v22, a9, a10);
      }
    }
    else
    {
      v26 = (__int64 *)sub_190AF50((__int64 ***)a2, (__int64)&v47, (__int64 *)a1);
      sub_164D160((__int64)a2, (__int64)v26, a3, a4, a5, a6, v27, v28, a9, a10);
      v29 = *((_BYTE *)v26 + 16);
      if ( v29 == 77 )
      {
        sub_14139C0(*(_QWORD *)a1, (__int64)v26);
        sub_164B7C0((__int64)v26, (__int64)a2);
        v29 = *((_BYTE *)v26 + 16);
      }
      if ( v29 > 0x17u )
      {
        v30 = a2[6];
        if ( v30 )
        {
          if ( a2[5] == v26[5] )
          {
            v40 = (__int64 *)a2[6];
            sub_1623A60((__int64)&v40, v30, 2);
            v31 = (__int64)(v26 + 6);
            if ( v26 + 6 == (__int64 *)&v40 )
            {
              if ( v40 )
                sub_161E7C0((__int64)&v40, (__int64)v40);
            }
            else
            {
              v32 = v26[6];
              if ( v32 )
              {
                sub_161E7C0((__int64)(v26 + 6), v32);
                v31 = (__int64)(v26 + 6);
              }
              v33 = (unsigned __int8 *)v40;
              v26[6] = (__int64)v40;
              if ( v33 )
                sub_1623210((__int64)&v40, v33, v31);
            }
          }
        }
      }
      v34 = *(_BYTE *)(*v26 + 8);
      if ( v34 == 16 )
        v34 = *(_BYTE *)(**(_QWORD **)(*v26 + 16) + 8LL);
      if ( v34 == 15 )
        sub_14134C0(*(_QWORD *)a1, v26);
      sub_190ACD0(a1 + 152, (__int64)a2);
      v36 = *(unsigned int *)(a1 + 680);
      if ( (unsigned int)v36 >= *(_DWORD *)(a1 + 684) )
      {
        sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, (int)&v40, v35);
        v36 = *(unsigned int *)(a1 + 680);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v36) = a2;
      v37 = *(__int64 **)(a1 + 104);
      ++*(_DWORD *)(a1 + 680);
      v39 = v26;
      v20 = 1;
      v40 = a2;
      sub_190E970(v37, (__int64 *)&v40, (__int64 *)&v39);
    }
  }
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  v23 = v44;
LABEL_14:
  if ( v23 != v46 )
    _libc_free((unsigned __int64)v23);
  return v20;
}
