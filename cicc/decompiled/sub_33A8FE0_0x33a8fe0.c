// Function: sub_33A8FE0
// Address: 0x33a8fe0
//
__int64 __fastcall sub_33A8FE0(__int64 a1, unsigned __int16 a2, __int64 a3)
{
  __int64 v5; // r13
  int v6; // eax
  __int64 v7; // r14
  _QWORD *v8; // rax
  __int64 *v9; // r14
  __int64 v10; // r13
  __int64 *v11; // rax
  int v12; // r14d
  __int64 v13; // r8
  __int64 v14; // r13
  _QWORD *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r8d
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // eax
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 *v35; // rax
  int v36; // [rsp+8h] [rbp-B8h]
  int v37; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+10h] [rbp-B0h]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  char v40; // [rsp+27h] [rbp-99h]
  int v41; // [rsp+28h] [rbp-98h]
  __int64 v42; // [rsp+30h] [rbp-90h] BYREF
  int v43; // [rsp+38h] [rbp-88h]
  __int128 v44; // [rsp+40h] [rbp-80h]
  __int64 v45; // [rsp+50h] [rbp-70h]
  __int64 v46; // [rsp+60h] [rbp-60h] BYREF
  __int64 v47; // [rsp+68h] [rbp-58h]
  __int64 v48; // [rsp+70h] [rbp-50h]
  __int64 v49; // [rsp+78h] [rbp-48h]
  __int64 v50; // [rsp+80h] [rbp-40h]
  __int64 v51; // [rsp+88h] [rbp-38h]

  if ( *(_BYTE *)a1 > 0x15u )
    goto LABEL_8;
  if ( (unsigned __int16)(a2 - 17) > 0xD3u )
  {
    if ( a2 > 1u && (unsigned __int16)(a2 - 504) > 7u )
    {
      v14 = *(_QWORD *)&byte_444C4A0[16 * a2 - 16];
      v15 = (_QWORD *)sub_BD5C60(a1);
      v10 = sub_BCD140(v15, v14);
      goto LABEL_14;
    }
LABEL_30:
    BUG();
  }
  v5 = a2 - 1;
  v6 = (unsigned __int16)word_4456580[v5];
  if ( (unsigned __int16)v6 <= 1u || (unsigned __int16)(v6 - 504) <= 7u )
    goto LABEL_30;
  v7 = *(_QWORD *)&byte_444C4A0[16 * v6 - 16];
  v8 = (_QWORD *)sub_BD5C60(a1);
  v9 = (__int64 *)sub_BCD140(v8, v7);
  if ( (unsigned __int16)(a2 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
  v10 = sub_BCDA70(v9, word_4456340[v5]);
LABEL_14:
  v16 = (_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a3 + 864) + 40LL));
  v17 = sub_9718F0(a1, v10, v16);
  if ( v17 )
    return sub_338B750(a3, v17);
LABEL_8:
  v11 = *(__int64 **)(a3 + 872);
  v12 = 0;
  if ( !v11
    || (v46 = a1,
        v47 = -1,
        v48 = 0,
        v49 = 0,
        v50 = 0,
        v51 = 0,
        (unsigned __int8)sub_CF4FA0(*v11, (__int64)&v46, (__int64)(v11 + 1), 0)) )
  {
    v19 = *(_QWORD *)(a3 + 864);
    v40 = 0;
    v13 = *(_QWORD *)(v19 + 384);
    v12 = *(_DWORD *)(v19 + 392);
  }
  else
  {
    v40 = 1;
    v13 = *(_QWORD *)(a3 + 864) + 288LL;
  }
  v37 = v13;
  v20 = sub_338B750(a3, a1);
  v46 = 0;
  v21 = v37;
  v22 = v20;
  v24 = v23;
  v25 = *(_QWORD *)(a3 + 864);
  v26 = *(_QWORD *)(a1 + 8);
  v47 = 0;
  v41 = v25;
  v48 = 0;
  v49 = 0;
  *(_QWORD *)&v44 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  LODWORD(v25) = *(unsigned __int8 *)(v26 + 8);
  *((_QWORD *)&v44 + 1) = 0;
  BYTE4(v45) = 0;
  if ( (unsigned int)(v25 - 17) <= 1 )
    v26 = **(_QWORD **)(v26 + 16);
  v27 = *(_DWORD *)(v26 + 8);
  v28 = *(_DWORD *)(a3 + 848);
  v42 = 0;
  v43 = v28;
  LODWORD(v45) = v27 >> 8;
  v29 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 )
  {
    if ( &v42 != (__int64 *)(v29 + 48) )
    {
      v30 = *(_QWORD *)(v29 + 48);
      v42 = v30;
      if ( v30 )
      {
        v36 = v37;
        v38 = v22;
        v39 = v24;
        sub_B96E90((__int64)&v42, v30, 1);
        v21 = v36;
        v22 = v38;
        v24 = v39;
      }
    }
  }
  v33 = sub_33F1F00(v41, a2, 0, (unsigned int)&v42, v21, v12, v22, v24, v44, v45, 256, 0, (__int64)&v46, 0);
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  if ( !v40 )
  {
    v34 = *(unsigned int *)(a3 + 136);
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 140) )
    {
      sub_C8D5F0(a3 + 128, (const void *)(a3 + 144), v34 + 1, 0x10u, v31, v32);
      v34 = *(unsigned int *)(a3 + 136);
    }
    v35 = (__int64 *)(*(_QWORD *)(a3 + 128) + 16 * v34);
    *v35 = v33;
    v35[1] = 1;
    ++*(_DWORD *)(a3 + 136);
  }
  return v33;
}
