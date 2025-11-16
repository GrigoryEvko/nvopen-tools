// Function: sub_18478A0
// Address: 0x18478a0
//
__int64 __fastcall sub_18478A0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  char v11; // al
  double v12; // xmm4_8
  double v13; // xmm5_8
  void **v15; // rax
  void **v16; // rbx
  void **v17; // rdx
  char v18[8]; // [rsp+70h] [rbp-140h] BYREF
  void **v19; // [rsp+78h] [rbp-138h]
  void **v20; // [rsp+80h] [rbp-130h]
  int v21; // [rsp+88h] [rbp-128h]
  int v22; // [rsp+8Ch] [rbp-124h]
  __int64 v23; // [rsp+B0h] [rbp-100h]
  unsigned __int64 v24; // [rsp+B8h] [rbp-F8h]
  int v25; // [rsp+C4h] [rbp-ECh]
  int v26; // [rsp+C8h] [rbp-E8h]
  char v27[8]; // [rsp+E0h] [rbp-D0h] BYREF
  int v28; // [rsp+E8h] [rbp-C8h] BYREF
  __int64 v29; // [rsp+F0h] [rbp-C0h]
  int *v30; // [rsp+F8h] [rbp-B8h]
  int *v31; // [rsp+100h] [rbp-B0h]
  __int64 v32; // [rsp+108h] [rbp-A8h]
  int v33; // [rsp+118h] [rbp-98h] BYREF
  __int64 v34; // [rsp+120h] [rbp-90h]
  int *v35; // [rsp+128h] [rbp-88h]
  int *v36; // [rsp+130h] [rbp-80h]
  __int64 v37; // [rsp+138h] [rbp-78h]
  int v38; // [rsp+148h] [rbp-68h] BYREF
  __int64 v39; // [rsp+150h] [rbp-60h]
  int *v40; // [rsp+158h] [rbp-58h]
  int *v41; // [rsp+160h] [rbp-50h]
  __int64 v42; // [rsp+168h] [rbp-48h]
  char v43; // [rsp+170h] [rbp-40h]

  v10 = 1;
  *(double *)a3.m128_u64 = (*(double (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
  v28 = 0;
  v30 = &v28;
  v31 = &v28;
  v35 = &v33;
  v36 = &v33;
  v40 = &v38;
  v41 = &v38;
  v29 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v42 = 0;
  v43 = v11;
  sub_18476F0((__int64)v18, (__int64)v27, a2, a3, a4, a5, a6, v12, v13, a9, a10);
  if ( v25 == v26 )
  {
    v15 = v19;
    if ( v20 == v19 )
    {
      v16 = &v19[v22];
      if ( v19 == v16 )
      {
        v17 = v19;
      }
      else
      {
        do
        {
          if ( *v15 == &unk_4F9EE48 )
            break;
          ++v15;
        }
        while ( v16 != v15 );
        v17 = &v19[v22];
      }
    }
    else
    {
      v16 = &v20[v21];
      v15 = (void **)sub_16CC9F0((__int64)v18, (__int64)&unk_4F9EE48);
      if ( *v15 == &unk_4F9EE48 )
      {
        if ( v20 == v19 )
          v17 = &v20[v22];
        else
          v17 = &v20[v21];
      }
      else
      {
        if ( v20 != v19 )
        {
          v15 = &v20[v21];
LABEL_11:
          LOBYTE(v10) = v16 == v15;
          goto LABEL_2;
        }
        v15 = &v20[v22];
        v17 = v15;
      }
    }
    while ( v17 != v15 && (unsigned __int64)*v15 >= 0xFFFFFFFFFFFFFFFELL )
      ++v15;
    goto LABEL_11;
  }
LABEL_2:
  if ( v24 != v23 )
    _libc_free(v24);
  if ( v20 != v19 )
    _libc_free((unsigned __int64)v20);
  j___libc_free_0(0);
  j___libc_free_0(0);
  j___libc_free_0(0);
  sub_18423B0(v39);
  sub_1842580(v34);
  sub_1842750(v29);
  return v10;
}
