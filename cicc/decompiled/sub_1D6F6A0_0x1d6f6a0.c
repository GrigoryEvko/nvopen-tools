// Function: sub_1D6F6A0
// Address: 0x1d6f6a0
//
__int64 __fastcall sub_1D6F6A0(__int64 a1, _QWORD *a2)
{
  __int64 (*v2)(); // rax
  unsigned int v3; // r14d
  __int64 v5; // rbx
  unsigned __int8 *v7; // rdi
  int *v8; // rax
  int i; // ecx
  int v10; // edx
  __int64 v11; // r15
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // r8
  _QWORD *v16; // r12
  __int64 v17; // rax
  int v18; // esi
  __int64 *v19; // r14
  int v20; // ecx
  __int64 v21; // rax
  _QWORD *v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r9
  _QWORD *v25; // rcx
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // r9
  int v29; // r10d
  _QWORD *v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+18h] [rbp-D8h]
  _QWORD *v32; // [rsp+20h] [rbp-D0h]
  __int64 v33; // [rsp+28h] [rbp-C8h]
  _QWORD *v34; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v35; // [rsp+3Fh] [rbp-B1h]
  __int64 v36; // [rsp+48h] [rbp-A8h] BYREF
  _QWORD v37[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+60h] [rbp-90h]
  __int64 *v39; // [rsp+70h] [rbp-80h] BYREF
  __int64 v40; // [rsp+78h] [rbp-78h]
  __int64 v41; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-68h]

  if ( !a1 )
    return 0;
  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 768LL);
  if ( v2 == sub_1D5A3E0 )
    return 0;
  v5 = a2[5];
  v35 = ((__int64 (__fastcall *)(__int64, _QWORD))v2)(a1, *a2);
  if ( !v35 )
    return 0;
  v7 = (unsigned __int8 *)*(a2 - 3);
  v39 = &v41;
  v40 = 0x1000000000LL;
  sub_15FAA20(v7, (__int64)&v39);
  if ( (_DWORD)v40 )
  {
    v8 = (int *)v39 + 1;
    for ( i = *(_DWORD *)v39; ; i = v10 )
    {
      if ( (int *)((char *)v39 + 4 * (unsigned int)(v40 - 1) + 4) == v8 )
        goto LABEL_13;
      v10 = *v8;
      if ( i != -1 && i != v10 && v10 != -1 )
        break;
      ++v8;
    }
    if ( v39 != &v41 )
      _libc_free((unsigned __int64)v39);
    return 0;
  }
LABEL_13:
  if ( v39 != &v41 )
    _libc_free((unsigned __int64)v39);
  v11 = a2[1];
  v39 = 0;
  v3 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  if ( v11 )
  {
    while ( 1 )
    {
      v16 = sub_1648700(v11);
      v17 = v16[5];
      v36 = v17;
      if ( v5 == v17 || (unsigned int)*((unsigned __int8 *)v16 + 16) - 47 > 2 )
        goto LABEL_20;
      v18 = v42;
      if ( !v42 )
        break;
      v12 = (v42 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v13 = (__int64 *)(v40 + 16LL * v12);
      v14 = *v13;
      if ( v17 != *v13 )
      {
        v29 = 1;
        v19 = 0;
        while ( v14 != -8 )
        {
          if ( !v19 && v14 == -16 )
            v19 = v13;
          v12 = (v42 - 1) & (v29 + v12);
          v13 = (__int64 *)(v40 + 16LL * v12);
          v14 = *v13;
          if ( v17 == *v13 )
            goto LABEL_18;
          ++v29;
        }
        if ( !v19 )
          v19 = v13;
        v39 = (__int64 *)((char *)v39 + 1);
        v20 = v41 + 1;
        if ( 4 * ((int)v41 + 1) < 3 * v42 )
        {
          if ( v42 - HIDWORD(v41) - v20 <= v42 >> 3 )
          {
LABEL_26:
            sub_1CD3B30((__int64)&v39, v18);
            sub_1CD3040((__int64)&v39, &v36, v37);
            v19 = (__int64 *)v37[0];
            v17 = v36;
            v20 = v41 + 1;
          }
          LODWORD(v41) = v20;
          if ( *v19 != -8 )
            --HIDWORD(v41);
          *v19 = v17;
          v19[1] = 0;
          goto LABEL_30;
        }
LABEL_25:
        v18 = 2 * v42;
        goto LABEL_26;
      }
LABEL_18:
      v15 = v13[1];
      if ( !v15 )
      {
        v19 = v13;
LABEL_30:
        v21 = sub_157EE30(v36);
        v22 = (_QWORD *)*(a2 - 9);
        v23 = *(a2 - 6);
        v24 = v21;
        v25 = (_QWORD *)*(a2 - 3);
        v38 = 257;
        v26 = v21 - 24;
        if ( v24 )
          v24 = v26;
        v30 = v22;
        v31 = v23;
        v32 = v25;
        v33 = v24;
        v27 = sub_1648A60(56, 3u);
        if ( v27 )
        {
          v28 = v33;
          v34 = v27;
          sub_15FA660((__int64)v27, v30, v31, v32, (__int64)v37, v28);
          v27 = v34;
        }
        v19[1] = (__int64)v27;
        v15 = (__int64)v27;
      }
      sub_1648780((__int64)v16, (__int64)a2, v15);
      v3 = v35;
LABEL_20:
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
      {
        if ( a2[1] )
          goto LABEL_36;
        goto LABEL_46;
      }
    }
    v39 = (__int64 *)((char *)v39 + 1);
    goto LABEL_25;
  }
LABEL_46:
  sub_15F20C0(a2);
  v3 = v35;
LABEL_36:
  j___libc_free_0(v40);
  return v3;
}
