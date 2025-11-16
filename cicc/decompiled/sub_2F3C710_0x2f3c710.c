// Function: sub_2F3C710
// Address: 0x2f3c710
//
__int64 __fastcall sub_2F3C710(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rax
  __int64 v3; // r15
  unsigned int v4; // r8d
  __int64 v5; // rax
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  unsigned int *v12; // rax
  int v13; // ecx
  unsigned int *v14; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r12
  __int16 v20; // ax
  _BYTE *v21; // rbx
  __int64 v22; // r12
  unsigned int *v23; // r12
  unsigned int *v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rbx
  __int64 v32; // [rsp-178h] [rbp-178h]
  __int16 v33; // [rsp-152h] [rbp-152h]
  __int64 v34; // [rsp-148h] [rbp-148h]
  _BYTE *v35[4]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v36; // [rsp-108h] [rbp-108h]
  __int64 v37[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v38; // [rsp-D8h] [rbp-D8h]
  unsigned int *v39; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v40; // [rsp-C0h] [rbp-C0h]
  _BYTE v41[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v42; // [rsp-98h] [rbp-98h]
  __int64 v43; // [rsp-90h] [rbp-90h]
  __int64 v44; // [rsp-88h] [rbp-88h]
  _QWORD *v45; // [rsp-80h] [rbp-80h]
  void **v46; // [rsp-78h] [rbp-78h]
  void **v47; // [rsp-70h] [rbp-70h]
  __int64 v48; // [rsp-68h] [rbp-68h]
  int v49; // [rsp-60h] [rbp-60h]
  __int16 v50; // [rsp-5Ch] [rbp-5Ch]
  char v51; // [rsp-5Ah] [rbp-5Ah]
  __int64 v52; // [rsp-58h] [rbp-58h]
  __int64 v53; // [rsp-50h] [rbp-50h]
  void *v54; // [rsp-48h] [rbp-48h] BYREF
  void *v55; // [rsp-40h] [rbp-40h] BYREF

  if ( !*(_QWORD *)(a1 + 16) )
    return 0;
  v1 = (_QWORD *)sub_B2BE50(a1);
  v2 = sub_BCB2D0(v1);
  v3 = *(_QWORD *)(a1 + 16);
  v4 = 0;
  v34 = v2;
  if ( v3 )
  {
    while ( 1 )
    {
      v5 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v6 = *(_QWORD *)(v5 + 24);
      if ( *(_BYTE *)v6 == 85 && a1 == *(_QWORD *)(v6 - 32) )
        break;
LABEL_4:
      if ( !v3 )
        return v4;
    }
    v7 = (_QWORD *)sub_BD5C60(v6);
    v42 = 0;
    v45 = v7;
    v48 = 0;
    v46 = &v54;
    v47 = &v55;
    v43 = 0;
    v49 = 0;
    v54 = &unk_49DA100;
    v50 = 512;
    v51 = 7;
    v52 = 0;
    v53 = 0;
    LOWORD(v44) = 0;
    v55 = &unk_49DA0B0;
    v8 = *(_QWORD *)(v6 + 40);
    v39 = (unsigned int *)v41;
    v42 = v8;
    v40 = 0x200000000LL;
    v43 = v6 + 24;
    v37[0] = *(_QWORD *)sub_B46C60(v6);
    if ( v37[0] && (sub_B96E90((__int64)v37, v37[0], 1), (v11 = v37[0]) != 0) )
    {
      v12 = v39;
      v13 = v40;
      v14 = &v39[4 * (unsigned int)v40];
      if ( v39 != v14 )
      {
        while ( *v12 )
        {
          v12 += 4;
          if ( v14 == v12 )
            goto LABEL_27;
        }
        *((_QWORD *)v12 + 1) = v37[0];
LABEL_14:
        sub_B91220((__int64)v37, v11);
LABEL_19:
        v38 = 257;
        v16 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
        v17 = *(_QWORD *)(v6 - 32 * v16);
        v35[0] = *(_BYTE **)(v6 + 32 * (1 - v16));
        v18 = sub_BCB2B0(v45);
        v19 = sub_921130(&v39, v18, v17, v35, 1, (__int64)v37, 0);
        HIBYTE(v20) = HIBYTE(v33);
        v36 = 257;
        v38 = 257;
        LOBYTE(v20) = 2;
        v33 = v20;
        v21 = sub_BD2C40(80, 1u);
        if ( v21 )
          sub_B4D190((__int64)v21, v34, v19, (__int64)v37, 0, v33, 0, 0);
        (*((void (__fastcall **)(void **, _BYTE *, _BYTE **, __int64, __int64))*v47 + 2))(v47, v21, v35, v43, v44);
        v22 = 4LL * (unsigned int)v40;
        if ( v39 != &v39[v22] )
        {
          v23 = &v39[v22];
          v24 = v39;
          do
          {
            v25 = *((_QWORD *)v24 + 1);
            v26 = *v24;
            v24 += 4;
            sub_B99FD0((__int64)v21, v26, v25);
          }
          while ( v23 != v24 );
        }
        v38 = 257;
        v27 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
        v35[0] = v21;
        v28 = sub_BCB2B0(v45);
        v29 = sub_921130(&v39, v28, v27, v35, 1, (__int64)v37, 0);
        sub_BD84D0(v6, v29);
        sub_B43D60((_QWORD *)v6);
        nullsub_61();
        v54 = &unk_49DA100;
        nullsub_63();
        if ( v39 != (unsigned int *)v41 )
          _libc_free((unsigned __int64)v39);
        v4 = 1;
        goto LABEL_4;
      }
LABEL_27:
      if ( (unsigned int)v40 >= (unsigned __int64)HIDWORD(v40) )
      {
        v30 = (unsigned int)v40 + 1LL;
        v31 = v32 & 0xFFFFFFFF00000000LL;
        v32 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v40) < v30 )
        {
          sub_C8D5F0((__int64)&v39, v41, v30, 0x10u, v9, v10);
          v14 = &v39[4 * (unsigned int)v40];
        }
        *(_QWORD *)v14 = v31;
        *((_QWORD *)v14 + 1) = v11;
        v11 = v37[0];
        LODWORD(v40) = v40 + 1;
      }
      else
      {
        if ( v14 )
        {
          *v14 = 0;
          *((_QWORD *)v14 + 1) = v11;
          v13 = v40;
          v11 = v37[0];
        }
        LODWORD(v40) = v13 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v39, 0);
      v11 = v37[0];
    }
    if ( !v11 )
      goto LABEL_19;
    goto LABEL_14;
  }
  return 0;
}
