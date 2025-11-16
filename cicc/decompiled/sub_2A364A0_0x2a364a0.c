// Function: sub_2A364A0
// Address: 0x2a364a0
//
__int64 __fastcall sub_2A364A0(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r9
  __int64 v11; // rax
  _QWORD *v12; // r8
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // rax
  _QWORD *v16; // r10
  _BYTE *v17; // rsi
  _BYTE *v18; // rsi
  int v19; // ecx
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rdi
  _BYTE *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int8 *v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int8 *v35; // rax
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  int v49; // eax
  _BYTE *v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // edi
  __int64 *v55; // [rsp+0h] [rbp-70h]
  __int64 *v56; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v57; // [rsp+10h] [rbp-60h]
  _QWORD *v58; // [rsp+10h] [rbp-60h]
  _QWORD *v59; // [rsp+10h] [rbp-60h]
  _QWORD *v60; // [rsp+10h] [rbp-60h]
  _QWORD *v61; // [rsp+10h] [rbp-60h]
  _QWORD *v62; // [rsp+10h] [rbp-60h]
  __int64 v63; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v65; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-58h]
  __int64 v67; // [rsp+18h] [rbp-58h]
  __int64 v68; // [rsp+18h] [rbp-58h]
  _QWORD *v69; // [rsp+18h] [rbp-58h]
  _QWORD *v70; // [rsp+18h] [rbp-58h]
  _QWORD *v71; // [rsp+18h] [rbp-58h]
  _QWORD *v72; // [rsp+18h] [rbp-58h]
  _QWORD *v73; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+20h] [rbp-50h]
  __int64 v76; // [rsp+20h] [rbp-50h]
  _QWORD v78[7]; // [rsp+38h] [rbp-38h] BYREF

  v10 = a6 + 56;
  v11 = *(_QWORD *)(a6 + 56);
  *(_QWORD *)(a6 + 136) += 160LL;
  v12 = (_QWORD *)((v11 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a6 + 64) >= (unsigned __int64)(v12 + 20) && v11 )
  {
    *(_QWORD *)(a6 + 56) = v12 + 20;
  }
  else
  {
    v67 = v10;
    v51 = sub_9D1E70(v10, 160, 160, 3);
    v10 = v67;
    v12 = (_QWORD *)v51;
  }
  memset(v12, 0, 0xA0u);
  v12[9] = 8;
  v12[8] = v12 + 11;
  *((_BYTE *)v12 + 84) = 1;
  v13 = *(_QWORD *)(a6 + 56);
  *(_QWORD *)(a6 + 136) += 160LL;
  v14 = (_QWORD *)((v13 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a6 + 64) >= (unsigned __int64)(v14 + 20) && v13 )
  {
    *(_QWORD *)(a6 + 56) = v14 + 20;
  }
  else
  {
    v58 = v12;
    v68 = v10;
    v52 = sub_9D1E70(v10, 160, 160, 3);
    v12 = v58;
    v10 = v68;
    v14 = (_QWORD *)v52;
  }
  memset(v14, 0, 0xA0u);
  v14[9] = 8;
  v14[8] = v14 + 11;
  *((_BYTE *)v14 + 84) = 1;
  v15 = *(_QWORD *)(a6 + 56);
  *(_QWORD *)(a6 + 136) += 160LL;
  v16 = (_QWORD *)((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a6 + 64) >= (unsigned __int64)(v16 + 20) && v15 )
  {
    *(_QWORD *)(a6 + 56) = v16 + 20;
  }
  else
  {
    v69 = v12;
    v53 = sub_9D1E70(v10, 160, 160, 3);
    v12 = v69;
    v16 = (_QWORD *)v53;
  }
  memset(v16, 0, 0xA0u);
  v16[9] = 8;
  v16[8] = v16 + 11;
  *((_BYTE *)v16 + 84) = 1;
  v78[0] = v16;
  *v16 = v14;
  v17 = (_BYTE *)v14[2];
  if ( v17 == (_BYTE *)v14[3] )
  {
    v60 = v16;
    v71 = v12;
    sub_D4C7F0((__int64)(v14 + 1), v17, v78);
    v16 = v60;
    v12 = v71;
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = v78[0];
      v17 = (_BYTE *)v14[2];
    }
    v14[2] = v17 + 8;
  }
  v78[0] = v14;
  *v14 = v12;
  v18 = (_BYTE *)v12[2];
  if ( v18 == (_BYTE *)v12[3] )
  {
    v59 = v16;
    v70 = v12;
    sub_D4C7F0((__int64)(v12 + 1), v18, v78);
    v16 = v59;
    v12 = v70;
  }
  else
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = v78[0];
      v18 = (_BYTE *)v12[2];
    }
    v12[2] = v18 + 8;
  }
  v19 = *(_DWORD *)(a6 + 24);
  v20 = *(_QWORD *)(a6 + 8);
  if ( v19 )
  {
    v21 = v19 - 1;
    v22 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (__int64 *)(v20 + 16LL * v22);
    v24 = *v23;
    if ( a2 == *v23 )
    {
LABEL_20:
      v25 = v23[1];
      if ( v25 )
      {
        v78[0] = v12;
        *v12 = v25;
        v26 = *(_BYTE **)(v25 + 16);
        if ( v26 == *(_BYTE **)(v25 + 24) )
        {
          v61 = v16;
          v72 = v12;
          sub_D4C7F0(v25 + 8, v26, v78);
          v12 = v72;
          v16 = v61;
        }
        else
        {
          if ( v26 )
          {
            *(_QWORD *)v26 = v78[0];
            v26 = *(_BYTE **)(v25 + 16);
          }
          *(_QWORD *)(v25 + 16) = v26 + 8;
        }
        goto LABEL_25;
      }
    }
    else
    {
      v49 = 1;
      while ( v24 != -4096 )
      {
        v54 = v49 + 1;
        v22 = v21 & (v49 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( a2 == *v23 )
          goto LABEL_20;
        v49 = v54;
      }
    }
  }
  v78[0] = v12;
  v50 = *(_BYTE **)(a6 + 40);
  if ( v50 == *(_BYTE **)(a6 + 48) )
  {
    v62 = v16;
    v73 = v12;
    sub_D4C7F0(a6 + 32, v50, v78);
    v16 = v62;
    v12 = v73;
  }
  else
  {
    if ( v50 )
    {
      *(_QWORD *)v50 = v12;
      v50 = *(_BYTE **)(a6 + 40);
    }
    *(_QWORD *)(a6 + 40) = v50 + 8;
  }
LABEL_25:
  v55 = v16;
  v56 = v12;
  v63 = a1[3];
  v27 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v57 = (unsigned __int8 *)sub_ACD640(v27, v63, 0);
  v64 = a1[1];
  v28 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v29 = (unsigned __int8 *)sub_ACD640(v28, v64, 0);
  v30 = sub_2A35A50(a2, a3, v29, v57, "cols", 4, (__int64 *)a4, a5, v56, a6);
  v31 = sub_AA56F0(v30);
  v32 = a1[3];
  *((_QWORD *)a1 + 7) = v31;
  v33 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v65 = (unsigned __int8 *)sub_ACD640(v33, v32, 0);
  v75 = *a1;
  v34 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v35 = (unsigned __int8 *)sub_ACD640(v34, v75, 0);
  v36 = sub_2A35A50(v30, *((_QWORD *)a1 + 7), v35, v65, "rows", 4, (__int64 *)a4, a5, v14, a6);
  v37 = sub_AA56F0(v36);
  v38 = a1[3];
  *((_QWORD *)a1 + 4) = v37;
  v39 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v66 = (unsigned __int8 *)sub_ACD640(v39, v38, 0);
  v76 = a1[2];
  v40 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
  v41 = (unsigned __int8 *)sub_ACD640(v40, v76, 0);
  v42 = sub_2A35A50(v36, *((_QWORD *)a1 + 4), v41, v66, "inner", 5, (__int64 *)a4, a5, v55, a6);
  *((_QWORD *)a1 + 10) = sub_AA56F0(v42);
  *((_QWORD *)a1 + 6) = sub_AA54C0(v30);
  *((_QWORD *)a1 + 3) = sub_AA54C0(v36);
  v43 = sub_AA54C0(v42);
  *((_QWORD *)a1 + 9) = v43;
  v44 = v43;
  v45 = *(_QWORD *)(*((_QWORD *)a1 + 3) + 56LL);
  if ( v45 )
    v45 -= 24;
  *((_QWORD *)a1 + 2) = v45;
  v46 = *(_QWORD *)(*((_QWORD *)a1 + 6) + 56LL);
  if ( v46 )
    v46 -= 24;
  *((_QWORD *)a1 + 5) = v46;
  v47 = *(_QWORD *)(v44 + 56);
  if ( v47 )
    v47 -= 24;
  *((_QWORD *)a1 + 8) = v47;
  return v42;
}
