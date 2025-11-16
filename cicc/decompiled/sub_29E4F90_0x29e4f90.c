// Function: sub_29E4F90
// Address: 0x29e4f90
//
void __fastcall sub_29E4F90(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r15
  char v13; // r13
  char v14; // r12
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  __int64 v20; // rsi
  _QWORD *v21; // rbx
  _QWORD *v22; // r12
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int8 *v27; // [rsp-180h] [rbp-180h]
  unsigned __int16 v28; // [rsp-172h] [rbp-172h]
  __int16 v29; // [rsp-170h] [rbp-170h]
  __int64 v30; // [rsp-170h] [rbp-170h]
  unsigned __int64 v31; // [rsp-168h] [rbp-168h]
  __int64 v32; // [rsp-160h] [rbp-160h]
  unsigned __int64 v33[2]; // [rsp-148h] [rbp-148h] BYREF
  __int64 v34; // [rsp-138h] [rbp-138h] BYREF
  _QWORD *v35; // [rsp-130h] [rbp-130h]
  __int64 v36; // [rsp-128h] [rbp-128h]
  _QWORD v37[8]; // [rsp-120h] [rbp-120h] BYREF
  __int64 v38; // [rsp-E0h] [rbp-E0h]
  char v39; // [rsp-D8h] [rbp-D8h]
  __int64 v40; // [rsp-D4h] [rbp-D4h]
  unsigned __int64 v41[2]; // [rsp-C8h] [rbp-C8h] BYREF
  _QWORD v42[6]; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v43; // [rsp-88h] [rbp-88h]
  __int64 v44; // [rsp-80h] [rbp-80h]
  void **v45; // [rsp-78h] [rbp-78h]
  void **v46; // [rsp-70h] [rbp-70h]
  __int64 v47; // [rsp-68h] [rbp-68h]
  int v48; // [rsp-60h] [rbp-60h]
  __int16 v49; // [rsp-5Ch] [rbp-5Ch]
  char v50; // [rsp-5Ah] [rbp-5Ah]
  __int64 v51; // [rsp-58h] [rbp-58h]
  __int64 v52; // [rsp-50h] [rbp-50h]
  void *v53; // [rsp-48h] [rbp-48h] BYREF
  void *v54; // [rsp-40h] [rbp-40h] BYREF

  if ( (_BYTE)qword_5009348 && *(_QWORD *)a2 )
  {
    v4 = sub_B491C0(a1);
    v5 = *(_QWORD *)(a2 + 8);
    v6 = v4;
    v32 = (*(__int64 (__fastcall **)(__int64, __int64))a2)(v5, v4);
    v7 = sub_B43CC0(a1);
    v10 = *(_QWORD *)(a1 - 32);
    v39 = 0;
    v31 = v7;
    v33[0] = (unsigned __int64)&v34;
    v33[1] = 0x100000000LL;
    v35 = v37;
    v36 = 0x600000000LL;
    v37[7] = 0;
    v38 = 0;
    v40 = 0;
    if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v10 + 24) )
      BUG();
    if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v10, v6, v8, v9);
      v11 = *(_QWORD *)(v10 + 96);
      v12 = v11 + 40LL * *(_QWORD *)(v10 + 104);
      if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v10, v6, v25, v26);
        v11 = *(_QWORD *)(v10 + 96);
      }
    }
    else
    {
      v11 = *(_QWORD *)(v10 + 96);
      v12 = v11 + 40LL * *(_QWORD *)(v10 + 104);
    }
    v13 = 0;
    while ( v12 != v11 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v11 + 8) + 8LL) == 14
          && !(unsigned __int8)sub_B2BAE0(v11)
          && !(unsigned __int8)sub_BD3610(v11, 0) )
        {
          v29 = sub_B2BD00(v11);
          v14 = HIBYTE(v29);
          if ( HIBYTE(v29) )
          {
            if ( !v13 )
            {
              v38 = sub_B491C0(a1);
              HIDWORD(v40) = *(_DWORD *)(v38 + 92);
              sub_B1F440((__int64)v33);
            }
            v13 = HIBYTE(v29);
            v15 = v28;
            BYTE1(v15) = 0;
            v27 = *(unsigned __int8 **)(a1
                                      + 32
                                      * (*(unsigned int *)(v11 + 32)
                                       - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
            v28 = (unsigned __int8)v28;
            if ( (unsigned __int8)sub_F518D0(v27, v15, v31, a1, v32, (__int64)v33) < (unsigned __int8)v29 )
              break;
          }
        }
        v11 += 40;
        if ( v12 == v11 )
          goto LABEL_21;
      }
      v44 = sub_BD5C60(a1);
      v49 = 512;
      v45 = &v53;
      v46 = &v54;
      v41[0] = (unsigned __int64)v42;
      v41[1] = 0x200000000LL;
      v43 = 0;
      v54 = &unk_49DA0B0;
      v47 = 0;
      v48 = 0;
      v50 = 7;
      v51 = 0;
      v52 = 0;
      v42[4] = 0;
      v42[5] = 0;
      v53 = &unk_49DA100;
      sub_D5F1F0((__int64)v41, a1);
      v30 = sub_B37DD0((__int64)v41, v31, (__int64)v27, 1LL << v29, 0);
      nullsub_61();
      v53 = &unk_49DA100;
      nullsub_63();
      v20 = v30;
      if ( (_QWORD *)v41[0] != v42 )
      {
        _libc_free(v41[0]);
        v20 = v30;
      }
      v11 += 40;
      v13 = v14;
      sub_CFEAE0(v32, v20, v16, v17, v18, v19);
    }
LABEL_21:
    v21 = v35;
    v22 = &v35[(unsigned int)v36];
    if ( v35 != v22 )
    {
      do
      {
        v23 = *--v22;
        if ( v23 )
        {
          v24 = *(_QWORD *)(v23 + 24);
          if ( v24 != v23 + 40 )
            _libc_free(v24);
          j_j___libc_free_0(v23);
        }
      }
      while ( v21 != v22 );
      v22 = v35;
    }
    if ( v22 != v37 )
      _libc_free((unsigned __int64)v22);
    if ( (__int64 *)v33[0] != &v34 )
      _libc_free(v33[0]);
  }
}
