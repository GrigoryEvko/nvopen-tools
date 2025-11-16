// Function: sub_2BFBAC0
// Address: 0x2bfbac0
//
__int64 *__fastcall sub_2BFBAC0(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v10; // r15
  __int64 *v11; // rbx
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v14; // rdi
  _BYTE *v15; // r14
  __int64 *v16; // rbx
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 *v30; // r12
  __int64 v31; // rdi
  __int64 (__fastcall *v32)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v33; // r15
  _QWORD *v35; // rax
  __int64 v36; // r9
  __int64 v37; // r12
  __int64 v38; // rbx
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  _QWORD *v48; // rax
  __int64 v49; // r12
  __int64 v50; // r13
  __int64 v51; // rbx
  __int64 v52; // rdx
  unsigned int v53; // esi
  _QWORD *v54; // rax
  __int64 v55; // r9
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // [rsp-10h] [rbp-100h]
  __int64 v61; // [rsp+0h] [rbp-F0h]
  int v62; // [rsp+24h] [rbp-CCh]
  unsigned __int8 *v63; // [rsp+28h] [rbp-C8h]
  __int64 v64; // [rsp+40h] [rbp-B0h]
  unsigned int v66; // [rsp+5Ch] [rbp-94h] BYREF
  _BYTE v67[32]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v68; // [rsp+80h] [rbp-70h]
  _BYTE v69[32]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v70; // [rsp+B0h] [rbp-40h]

  v61 = a2;
  v64 = sub_2BFB120(a1, a2, a3);
  v4 = sub_2BFB640(a1, a2, 0);
  v63 = (unsigned __int8 *)sub_2BF0180(a3, *(_QWORD *)(a1 + 904), (__int64 *)(a1 + 8));
  v5 = *(_QWORD *)(v4 + 8);
  if ( *(_BYTE *)(v5 + 8) != 15 )
  {
    v68 = 257;
    v30 = *(__int64 **)(a1 + 904);
    v31 = v30[10];
    v32 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v31 + 104LL);
    if ( v32 == sub_948040 )
    {
      if ( *(_BYTE *)v4 > 0x15u || *(_BYTE *)v64 > 0x15u || *v63 > 0x15u )
        goto LABEL_69;
      v33 = sub_AD5A90(v4, (_BYTE *)v64, v63, 0);
    }
    else
    {
      v33 = v32(v31, (_BYTE *)v4, (_BYTE *)v64, v63);
    }
    if ( v33 )
    {
LABEL_43:
      v4 = v33;
      return sub_2BF26E0(a1, v61, v4, 0);
    }
LABEL_69:
    v70 = 257;
    v54 = sub_BD2C40(72, 3u);
    v33 = (__int64)v54;
    if ( v54 )
      sub_B4DFA0((__int64)v54, v4, v64, (__int64)v63, (__int64)v69, v55, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v30[11] + 16LL))(
      v30[11],
      v33,
      v67,
      v30[7],
      v30[8]);
    v56 = *v30;
    v57 = *v30 + 16LL * *((unsigned int *)v30 + 2);
    while ( v57 != v56 )
    {
      v58 = *(_QWORD *)(v56 + 8);
      v59 = *(_DWORD *)v56;
      v56 += 16;
      sub_B99FD0(v33, v59, v58);
    }
    goto LABEL_43;
  }
  v66 = 0;
  v62 = *(_DWORD *)(v5 + 12);
  if ( v62 )
  {
    while ( 1 )
    {
      v20 = *(__int64 **)(a1 + 904);
      v68 = 257;
      v21 = v20[10];
      v22 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v21 + 80LL);
      if ( v22 != sub_92FAE0 )
        break;
      if ( *(_BYTE *)v64 <= 0x15u )
      {
        v6 = sub_AAADB0(v64, &v66, 1);
        goto LABEL_5;
      }
LABEL_27:
      v70 = 257;
      v6 = (__int64)sub_BD2C40(104, 1u);
      if ( v6 )
      {
        v23 = sub_B501B0(*(_QWORD *)(v64 + 8), &v66, 1);
        sub_B44260(v6, v23, 64, 1u, 0, 0);
        if ( *(_QWORD *)(v6 - 32) )
        {
          v24 = *(_QWORD *)(v6 - 24);
          **(_QWORD **)(v6 - 16) = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = *(_QWORD *)(v6 - 16);
        }
        *(_QWORD *)(v6 - 32) = v64;
        v25 = *(_QWORD *)(v64 + 16);
        *(_QWORD *)(v6 - 24) = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = v6 - 24;
        *(_QWORD *)(v6 - 16) = v64 + 16;
        *(_QWORD *)(v64 + 16) = v6 - 32;
        *(_QWORD *)(v6 + 72) = v6 + 88;
        *(_QWORD *)(v6 + 80) = 0x400000000LL;
        sub_B50030(v6, &v66, 1, (__int64)v69);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v20[11] + 16LL))(
        v20[11],
        v6,
        v67,
        v20[7],
        v20[8]);
      v26 = *v20;
      v27 = *v20 + 16LL * *((unsigned int *)v20 + 2);
      while ( v27 != v26 )
      {
        v28 = *(_QWORD *)(v26 + 8);
        v29 = *(_DWORD *)v26;
        v26 += 16;
        sub_B99FD0(v6, v29, v28);
      }
LABEL_6:
      v7 = *(__int64 **)(a1 + 904);
      v68 = 257;
      v8 = v7[10];
      v9 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v8 + 80LL);
      if ( v9 != sub_92FAE0 )
      {
        v10 = v9(v8, (_BYTE *)v4, (__int64)&v66, 1);
LABEL_9:
        if ( v10 )
          goto LABEL_10;
        goto LABEL_50;
      }
      if ( *(_BYTE *)v4 <= 0x15u )
      {
        v10 = sub_AAADB0(v4, &v66, 1);
        goto LABEL_9;
      }
LABEL_50:
      v70 = 257;
      v10 = (__int64)sub_BD2C40(104, 1u);
      if ( v10 )
      {
        v41 = sub_B501B0(*(_QWORD *)(v4 + 8), &v66, 1);
        sub_B44260(v10, v41, 64, 1u, 0, 0);
        if ( *(_QWORD *)(v10 - 32) )
        {
          v42 = *(_QWORD *)(v10 - 24);
          **(_QWORD **)(v10 - 16) = v42;
          if ( v42 )
            *(_QWORD *)(v42 + 16) = *(_QWORD *)(v10 - 16);
        }
        *(_QWORD *)(v10 - 32) = v4;
        v43 = *(_QWORD *)(v4 + 16);
        *(_QWORD *)(v10 - 24) = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 16) = v10 - 24;
        *(_QWORD *)(v10 - 16) = v4 + 16;
        *(_QWORD *)(v4 + 16) = v10 - 32;
        *(_QWORD *)(v10 + 72) = v10 + 88;
        *(_QWORD *)(v10 + 80) = 0x400000000LL;
        sub_B50030(v10, &v66, 1, (__int64)v69);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v7[11] + 16LL))(
        v7[11],
        v10,
        v67,
        v7[7],
        v7[8]);
      v44 = *v7;
      v45 = *v7 + 16LL * *((unsigned int *)v7 + 2);
      while ( v45 != v44 )
      {
        v46 = *(_QWORD *)(v44 + 8);
        v47 = *(_DWORD *)v44;
        v44 += 16;
        sub_B99FD0(v10, v47, v46);
      }
LABEL_10:
      v68 = 257;
      v11 = *(__int64 **)(a1 + 904);
      v12 = v11[10];
      v13 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v12 + 104LL);
      if ( v13 != sub_948040 )
      {
        v15 = (_BYTE *)v13(v12, (_BYTE *)v10, (_BYTE *)v6, v63);
LABEL_17:
        if ( v15 )
          goto LABEL_18;
        goto LABEL_45;
      }
      v14 = 0;
      if ( *(_BYTE *)v10 <= 0x15u )
        v14 = v10;
      if ( *(_BYTE *)v6 <= 0x15u && *v63 <= 0x15u && v14 )
      {
        v15 = (_BYTE *)sub_AD5A90(v14, (_BYTE *)v6, v63, 0);
        goto LABEL_17;
      }
LABEL_45:
      v70 = 257;
      v35 = sub_BD2C40(72, 3u);
      v36 = 0;
      v15 = v35;
      if ( v35 )
      {
        sub_B4DFA0((__int64)v35, v10, v6, (__int64)v63, (__int64)v69, 0, 0, 0);
        v36 = v60;
      }
      (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
        v11[11],
        v15,
        v67,
        v11[7],
        v11[8],
        v36,
        v61);
      v37 = *v11;
      v38 = *v11 + 16LL * *((unsigned int *)v11 + 2);
      while ( v38 != v37 )
      {
        v39 = *(_QWORD *)(v37 + 8);
        v40 = *(_DWORD *)v37;
        v37 += 16;
        sub_B99FD0((__int64)v15, v40, v39);
      }
LABEL_18:
      v68 = 257;
      v16 = *(__int64 **)(a1 + 904);
      v17 = v16[10];
      v18 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v17 + 88LL);
      if ( v18 == sub_9482E0 )
      {
        if ( *(_BYTE *)v4 > 0x15u || *v15 > 0x15u )
        {
LABEL_60:
          v70 = 257;
          v48 = sub_BD2C40(104, unk_3F148BC);
          v49 = (__int64)v48;
          if ( v48 )
          {
            sub_B44260((__int64)v48, *(_QWORD *)(v4 + 8), 65, 2u, 0, 0);
            *(_QWORD *)(v49 + 72) = v49 + 88;
            *(_QWORD *)(v49 + 80) = 0x400000000LL;
            sub_B4FD20(v49, v4, (__int64)v15, &v66, 1, (__int64)v69);
          }
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
            v16[11],
            v49,
            v67,
            v16[7],
            v16[8]);
          v50 = *v16;
          v51 = *v16 + 16LL * *((unsigned int *)v16 + 2);
          while ( v51 != v50 )
          {
            v52 = *(_QWORD *)(v50 + 8);
            v53 = *(_DWORD *)v50;
            v50 += 16;
            sub_B99FD0(v49, v53, v52);
          }
          v4 = v49;
          goto LABEL_24;
        }
        v19 = sub_AAAE30(v4, (__int64)v15, &v66, 1);
      }
      else
      {
        v19 = v18(v17, (_BYTE *)v4, v15, (__int64)&v66, 1);
      }
      if ( !v19 )
        goto LABEL_60;
      v4 = v19;
LABEL_24:
      if ( ++v66 == v62 )
        return sub_2BF26E0(a1, v61, v4, 0);
    }
    v6 = v22(v21, (_BYTE *)v64, (__int64)&v66, 1);
LABEL_5:
    if ( v6 )
      goto LABEL_6;
    goto LABEL_27;
  }
  return sub_2BF26E0(a1, v61, v4, 0);
}
