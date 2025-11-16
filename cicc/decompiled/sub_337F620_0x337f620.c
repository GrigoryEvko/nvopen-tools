// Function: sub_337F620
// Address: 0x337f620
//
void __fastcall sub_337F620(__int64 a1, __int64 a2, __int128 **a3)
{
  __int64 v4; // r12
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  int v12; // eax
  int v13; // r15d
  __int64 v14; // rax
  int v15; // eax
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rax
  __int128 *v18; // r15
  __int64 v19; // rdi
  unsigned __int16 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int128 *v24; // r13
  int v25; // eax
  int v26; // edx
  __int64 v27; // rax
  int v28; // edx
  __int64 v29; // r8
  __int64 v30; // r13
  int v31; // r15d
  _QWORD *v32; // rax
  __int64 v33; // rsi
  unsigned __int16 v34; // dx
  __int64 v35; // r9
  bool v36; // al
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r8
  unsigned __int16 v40; // ax
  __int64 v41; // rdx
  __int128 v42; // [rsp+0h] [rbp-F0h]
  __int64 v43; // [rsp+18h] [rbp-D8h]
  __int64 v44; // [rsp+18h] [rbp-D8h]
  __int64 v45; // [rsp+20h] [rbp-D0h]
  __int64 v46; // [rsp+60h] [rbp-90h] BYREF
  int v47; // [rsp+68h] [rbp-88h]
  __int64 v48; // [rsp+70h] [rbp-80h] BYREF
  __int64 v49; // [rsp+78h] [rbp-78h]
  __int64 v50; // [rsp+80h] [rbp-70h] BYREF
  __int64 v51; // [rsp+88h] [rbp-68h]
  __int64 v52; // [rsp+90h] [rbp-60h]
  __int64 v53[10]; // [rsp+A0h] [rbp-50h] BYREF

  v4 = a2;
  v6 = *(_DWORD *)(a1 + 848);
  v7 = *(_QWORD *)a1;
  v46 = 0;
  v47 = v6;
  if ( v7 )
  {
    if ( &v46 != (__int64 *)(v7 + 48) )
    {
      a2 = *(_QWORD *)(v7 + 48);
      v46 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v46, a2, 1);
    }
  }
  v8 = *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  v9 = *(_QWORD *)(*(_QWORD *)*a3 + 48LL) + 16LL * *((unsigned int *)*a3 + 2);
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v48) = v10;
  v49 = v11;
  LOWORD(v12) = sub_B5A5E0(v4);
  v13 = v12;
  if ( !BYTE1(v12) )
  {
    v34 = v48;
    v35 = *(_QWORD *)(a1 + 864);
    if ( (_WORD)v48 )
    {
      if ( (unsigned __int16)(v48 - 17) <= 0xD3u )
      {
        v39 = 0;
        v34 = word_4456580[(unsigned __int16)v48 - 1];
        goto LABEL_19;
      }
    }
    else
    {
      v44 = *(_QWORD *)(a1 + 864);
      v36 = sub_30070B0((__int64)&v48);
      v35 = v44;
      v34 = 0;
      if ( v36 )
      {
        v40 = sub_3009970((__int64)&v48, a2, 0, v37, v38);
        v35 = v44;
        v39 = v41;
        v34 = v40;
        goto LABEL_19;
      }
    }
    v39 = v49;
LABEL_19:
    LOBYTE(v13) = sub_33CC4A0(v35, v34, v39);
  }
  sub_B91FC0(v53, v4);
  v14 = *(_QWORD *)(v8 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  v15 = *(_DWORD *)(v14 + 8) >> 8;
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  v51 = 0;
  LODWORD(v52) = v15;
  BYTE4(v52) = 0;
  v50 = 0;
  v17 = sub_2E7BD70(v16, 2u, -1, v13, (int)v53, 0, 0, v52, 1u, 0, 0);
  v18 = *a3;
  v43 = v17;
  v19 = *(_QWORD *)(a1 + 864);
  v20 = (unsigned __int16 *)(*(_QWORD *)(*((_QWORD *)*a3 + 2) + 48LL) + 16LL * *((unsigned int *)*a3 + 6));
  v21 = *((_QWORD *)v20 + 1);
  v22 = *v20;
  v50 = 0;
  LODWORD(v51) = 0;
  *(_QWORD *)&v42 = sub_33F17F0(v19, 51, &v50, v22, v21);
  *((_QWORD *)&v42 + 1) = v23;
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
  v24 = *a3;
  v25 = sub_33738A0(a1);
  v27 = sub_33F5F90(
          v19,
          v25,
          v26,
          (unsigned int)&v46,
          *(_QWORD *)v24,
          *((_QWORD *)v24 + 1),
          *((_QWORD *)v24 + 2),
          *((_QWORD *)v24 + 3),
          v42,
          v18[2],
          v18[3],
          v18[4],
          v48,
          v49,
          v43,
          0,
          0,
          0);
  v29 = *(_QWORD *)(a1 + 864);
  v30 = v27;
  v31 = v28;
  if ( v27 )
  {
    v45 = *(_QWORD *)(a1 + 864);
    nullsub_1875(v27, v45, 0);
    *(_QWORD *)(v45 + 384) = v30;
    *(_DWORD *)(v45 + 392) = v31;
    sub_33E2B60(v45, 0);
  }
  else
  {
    *(_QWORD *)(v29 + 384) = 0;
    *(_DWORD *)(v29 + 392) = v28;
  }
  v50 = v4;
  v32 = sub_337DC20(a1 + 8, &v50);
  *v32 = v30;
  v33 = v46;
  *((_DWORD *)v32 + 2) = v31;
  if ( v33 )
    sub_B91220((__int64)&v46, v33);
}
