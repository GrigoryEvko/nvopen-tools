// Function: sub_33BFD70
// Address: 0x33bfd70
//
void __fastcall sub_33BFD70(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r13
  bool v15; // zf
  __int64 v16; // rsi
  __int64 v17; // rdx
  _BYTE *v18; // r8
  __int64 *v19; // rax
  char v20; // dl
  __int64 v21; // r14
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // r9d
  __int64 v29; // rdx
  __int128 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rbx
  int v36; // r12d
  __int128 v37; // [rsp-10h] [rbp-100h]
  __int64 v38; // [rsp+8h] [rbp-E8h]
  __int64 v39; // [rsp+18h] [rbp-D8h]
  __int64 v40; // [rsp+20h] [rbp-D0h]
  __int64 v41; // [rsp+20h] [rbp-D0h]
  __int64 v42; // [rsp+50h] [rbp-A0h] BYREF
  int v43; // [rsp+58h] [rbp-98h]
  __int64 v44; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v45; // [rsp+68h] [rbp-88h]
  __int64 v46; // [rsp+70h] [rbp-80h]
  int v47; // [rsp+78h] [rbp-78h]
  char v48; // [rsp+7Ch] [rbp-74h]
  __int64 v49; // [rsp+80h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL);
  sub_338BA40((__int64 *)a1, (int *)a2, 0);
  sub_33BFCB0(a1, a2);
  v48 = 1;
  v45 = &v49;
  v4 = *(unsigned int *)(a2 + 88);
  v5 = *(_QWORD *)(a1 + 960);
  v46 = 0x100000008LL;
  v47 = 0;
  v44 = 1;
  v6 = *(_QWORD *)(a2 - 32 * v4 - 64);
  v7 = *(unsigned int *)(v6 + 44);
  v49 = v6;
  v38 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 8 * v7);
  sub_3373E10(a1, v3, v38, 0x80000000LL, v8, v9);
  v12 = *(unsigned int *)(a2 + 88);
  if ( (_DWORD)v12 )
  {
    v40 = *(unsigned int *)(a2 + 88);
    v13 = a2 - 32;
    v14 = 0;
    while ( 1 )
    {
      v15 = v48 == 0;
      v16 = *(_QWORD *)(v13 + 32 * (v14 - v12));
      v17 = *(unsigned int *)(v16 + 44);
      v18 = *(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 56LL) + 8 * v17);
      v18[262] = 1;
      v18[217] = 1;
      v18[232] = 1;
      if ( v15 )
        goto LABEL_10;
      v19 = v45;
      v10 = HIDWORD(v46);
      v17 = (__int64)&v45[HIDWORD(v46)];
      if ( v45 != (__int64 *)v17 )
      {
        while ( v16 != *v19 )
        {
          if ( (__int64 *)v17 == ++v19 )
            goto LABEL_23;
        }
LABEL_8:
        if ( v40 == ++v14 )
          break;
        goto LABEL_9;
      }
LABEL_23:
      if ( HIDWORD(v46) >= (unsigned int)v46 )
      {
LABEL_10:
        v39 = (__int64)v18;
        sub_C8CC70((__int64)&v44, v16, v17, v10, (__int64)v18, v11);
        if ( !v20 )
          goto LABEL_8;
        sub_3373E10(a1, v3, v39, 0, v39, v11);
      }
      else
      {
        ++HIDWORD(v46);
        *(_QWORD *)v17 = v16;
        ++v44;
        sub_3373E10(a1, v3, (__int64)v18, 0, (__int64)v18, v11);
      }
      if ( v40 == ++v14 )
        break;
LABEL_9:
      v12 = *(unsigned int *)(a2 + 88);
    }
  }
  sub_2E33470(*(unsigned int **)(v3 + 144), *(unsigned int **)(v3 + 152));
  v21 = *(_QWORD *)(a1 + 864);
  v22 = sub_33EEAD0(v21, v38);
  v24 = v23;
  v42 = 0;
  *(_QWORD *)&v30 = sub_3373A60(a1, v38, v23, v25, v26, v27);
  *((_QWORD *)&v30 + 1) = v29;
  v31 = *(_QWORD *)a1;
  v43 = *(_DWORD *)(a1 + 848);
  if ( v31 )
  {
    if ( &v42 != (__int64 *)(v31 + 48) )
    {
      v32 = *(_QWORD *)(v31 + 48);
      v42 = v32;
      if ( v32 )
      {
        v41 = v30;
        sub_B96E90((__int64)&v42, v32, 1);
        *(_QWORD *)&v30 = v41;
      }
    }
  }
  *((_QWORD *)&v37 + 1) = v24;
  *(_QWORD *)&v37 = v22;
  v33 = sub_3406EB0(v21, 301, (unsigned int)&v42, 1, 0, v28, v30, v37);
  v35 = v33;
  v36 = v34;
  if ( v33 )
  {
    nullsub_1875(v33, v21, 0);
    *(_QWORD *)(v21 + 384) = v35;
    *(_DWORD *)(v21 + 392) = v36;
    sub_33E2B60(v21, 0);
  }
  else
  {
    *(_QWORD *)(v21 + 384) = 0;
    *(_DWORD *)(v21 + 392) = v34;
  }
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  if ( !v48 )
    _libc_free((unsigned __int64)v45);
}
