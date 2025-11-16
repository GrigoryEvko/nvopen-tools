// Function: sub_2E315B0
// Address: 0x2e315b0
//
__int64 __fastcall sub_2E315B0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  unsigned int v3; // r12d
  int v6; // r15d
  __int64 (*v7)(); // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r11
  __int64 (*v16)(void); // rax
  __int64 *v17; // rax
  __int64 *v18; // rsi
  __int64 *v19; // r14
  __int64 v20; // r12
  int v21; // eax
  __int64 v22; // r10
  __int64 (*v23)(); // rax
  char v24; // al
  __int64 v25; // [rsp+0h] [rbp-120h]
  unsigned __int8 v26; // [rsp+17h] [rbp-109h]
  __int64 v27; // [rsp+18h] [rbp-108h]
  __int64 v28; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v29; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v30; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+48h] [rbp-D8h]
  _BYTE v32[208]; // [rsp+50h] [rbp-D0h] BYREF

  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(_QWORD *)(v1 + 8);
  v3 = *(_BYTE *)(v2 + 688) & 1;
  if ( (*(_BYTE *)(v2 + 688) & 1) != 0 )
    return 0;
  v6 = sub_2E31540(a1);
  if ( v6 < 0 )
    goto LABEL_5;
  v11 = *(_QWORD *)(*(_QWORD *)(v1 + 64) + 8LL) + 32LL * v6;
  v12 = *(__int64 **)v11;
  v13 = *(__int64 **)(v11 + 8);
  if ( v12 == v13 )
    goto LABEL_5;
  v14 = *v12;
  if ( !*v12 )
  {
    while ( v13 != ++v12 )
    {
      v14 = *v12;
      if ( *v12 )
        goto LABEL_16;
    }
LABEL_5:
    v7 = *(__int64 (**)())(**(_QWORD **)(v1 + 16) + 128LL);
    if ( v7 == sub_2DAC790 )
    {
      v28 = 0;
      v29 = 0;
      v30 = v32;
      v31 = 0x400000000LL;
      BUG();
    }
    v8 = v7();
    v28 = 0;
    v29 = 0;
    v9 = v8;
    v30 = v32;
    v31 = 0x400000000LL;
    v10 = *(__int64 (**)())(*(_QWORD *)v8 + 344LL);
    if ( v10 != sub_2DB1AE0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v10)(
              v9,
              a1,
              &v28,
              &v29,
              &v30,
              0) )
      {
        v3 = 1;
        if ( v28 )
          LOBYTE(v3) = v29 != v28;
      }
      if ( v30 != v32 )
        _libc_free((unsigned __int64)v30);
    }
    return v3;
  }
LABEL_16:
  v15 = 0;
  v16 = *(__int64 (**)(void))(**(_QWORD **)(v1 + 16) + 128LL);
  if ( v16 != sub_2DAC790 )
    v15 = v16();
  v30 = v32;
  v31 = 0x400000000LL;
  v17 = *(__int64 **)(v14 + 64);
  v18 = &v17[*(unsigned int *)(v14 + 72)];
  if ( v17 == v18 )
  {
    return 1;
  }
  else
  {
    v27 = v1;
    v26 = v3;
    v19 = *(__int64 **)(v14 + 64);
    v20 = v15;
    do
    {
      v22 = *v19;
      if ( a1 != *v19 )
      {
        v28 = 0;
        v29 = 0;
        LODWORD(v31) = 0;
        v23 = *(__int64 (**)())(*(_QWORD *)v20 + 344LL);
        if ( v23 == sub_2DB1AE0
          || (v25 = v22,
              v24 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v23)(
                      v20,
                      v22,
                      &v28,
                      &v29,
                      &v30,
                      0),
              v22 = v25,
              v24) )
        {
          v21 = sub_2E31540(v22);
          if ( v21 < 0 || v6 == v21 )
          {
            v1 = v27;
            v3 = v26;
            if ( v30 != v32 )
              _libc_free((unsigned __int64)v30);
            goto LABEL_5;
          }
        }
      }
      ++v19;
    }
    while ( v18 != v19 );
    if ( v30 == v32 )
      return 1;
    _libc_free((unsigned __int64)v30);
    return 1;
  }
}
