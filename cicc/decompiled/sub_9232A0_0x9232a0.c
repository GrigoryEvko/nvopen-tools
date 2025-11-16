// Function: sub_9232A0
// Address: 0x9232a0
//
__int64 __fastcall sub_9232A0(
        __int64 a1,
        _DWORD *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        int a9,
        int a10,
        __int64 a11,
        int a12,
        char a13)
{
  __int64 v13; // rbx
  __int64 v14; // r13
  unsigned int v15; // esi
  __int64 v16; // r10
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 i; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  int v24; // edx
  int v25; // r8d
  __int64 v26; // rdi
  __int64 v27; // rax
  unsigned __int8 v28; // al
  __int64 v29; // rax
  __int64 v30; // r14
  unsigned int *v31; // r13
  unsigned int *v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // rsi
  int v35; // eax
  int v36; // edx
  int v37; // eax
  __int64 v39; // rdx
  unsigned int v40; // r14d
  __int64 v41; // rax
  __int64 v42; // r14
  unsigned int *v43; // r14
  unsigned int *v44; // rbx
  __int64 v45; // rdx
  __int64 v46; // rsi
  bool v47; // al
  __int64 v48; // rax
  int v50; // [rsp+10h] [rbp-E0h]
  int v51; // [rsp+18h] [rbp-D8h]
  int v52; // [rsp+1Ch] [rbp-D4h]
  int v54; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v55; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v56; // [rsp+38h] [rbp-B8h]
  __int64 v57; // [rsp+38h] [rbp-B8h]
  __int64 v58; // [rsp+40h] [rbp-B0h]
  __int64 v59; // [rsp+40h] [rbp-B0h]
  __int64 v60; // [rsp+48h] [rbp-A8h]
  _BYTE *v61; // [rsp+58h] [rbp-98h] BYREF
  _QWORD v62[4]; // [rsp+60h] [rbp-90h] BYREF
  char v63; // [rsp+80h] [rbp-70h]
  char v64; // [rsp+81h] [rbp-6Fh]
  _QWORD v65[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v66; // [rsp+B0h] [rbp-40h]

  v13 = a1;
  if ( a7 != 1 )
    sub_91B8A0("error generating code for loading from bitfield!", a2, 1);
  v14 = sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a11 + 120), (__int64)a3, a4);
  v15 = *(_DWORD *)(*(_QWORD *)(a8 + 8) + 8LL);
  v64 = 1;
  v60 = a1 + 48;
  v62[0] = "tmp";
  v63 = 3;
  v16 = sub_BCE760(v14, v15 >> 8);
  if ( v16 == *(_QWORD *)(a8 + 8) )
  {
    v20 = a8;
    goto LABEL_10;
  }
  v17 = *(_QWORD *)(a1 + 128);
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v17 + 120LL);
  if ( v18 != sub_920130 )
  {
    v59 = v16;
    v48 = v18(v17, 49u, (_BYTE *)a8, v16);
    v16 = v59;
    v20 = v48;
    goto LABEL_9;
  }
  if ( *(_BYTE *)a8 <= 0x15u )
  {
    v58 = v16;
    if ( (unsigned __int8)sub_AC4810(49) )
      v19 = sub_ADAB70(49, a8, v58, 0);
    else
      v19 = sub_AA93C0(49, a8, v58);
    v16 = v58;
    v20 = v19;
LABEL_9:
    if ( v20 )
      goto LABEL_10;
  }
  v66 = 257;
  v20 = sub_B51D30(49, a8, v16, v65, 0, 0);
  if ( (unsigned __int8)sub_920620(v20) )
  {
    v39 = *(_QWORD *)(v13 + 144);
    v40 = *(_DWORD *)(v13 + 152);
    if ( v39 )
      sub_B99FD0(v20, 3, v39);
    sub_B45150(v20, v40);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v13 + 136) + 16LL))(
    *(_QWORD *)(v13 + 136),
    v20,
    v62,
    *(_QWORD *)(v60 + 56),
    *(_QWORD *)(v60 + 64));
  v41 = *(_QWORD *)(v13 + 48);
  v42 = 16LL * *(unsigned int *)(v13 + 56);
  if ( v41 != v41 + v42 )
  {
    v57 = v13;
    v43 = (unsigned int *)(v41 + v42);
    v44 = *(unsigned int **)(v13 + 48);
    do
    {
      v45 = *((_QWORD *)v44 + 1);
      v46 = *v44;
      v44 += 4;
      sub_B99FD0(v20, v46, v45);
    }
    while ( v43 != v44 );
    v13 = v57;
  }
LABEL_10:
  for ( i = *(_QWORD *)(a11 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v55 = *(_QWORD *)(i + 128);
  v56 = *(_QWORD *)(a11 + 128) / v55;
  v22 = sub_BCB2D0(*(_QWORD *)(v13 + 40));
  v61 = (_BYTE *)sub_ACD640(v22, v56, 0);
  v65[0] = "tmp";
  v66 = 259;
  v23 = sub_921130((unsigned int **)v60, v14, v20, &v61, 1, (__int64)v65, 0);
  v24 = v23;
  if ( a3 )
    *a3 = v23;
  if ( unk_4D0463C && (v54 = v23, v47 = sub_90AA40(*(_QWORD *)(v13 + 32), v23), v24 = v54, v47) )
    v25 = 1;
  else
    v25 = a13 & 1;
  v26 = *(_QWORD *)(v13 + 96);
  v64 = 1;
  v51 = v25;
  v50 = v24;
  v62[0] = "tmp";
  v63 = 3;
  v27 = sub_AA4E30(v26);
  v28 = sub_AE5020(v27, v14);
  v66 = 257;
  v52 = v28;
  v29 = sub_BD2C40(80, unk_3F10A14);
  v30 = v29;
  if ( v29 )
    sub_B4D190(v29, v14, v50, (unsigned int)v65, v51, v52, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v13 + 136) + 16LL))(
    *(_QWORD *)(v13 + 136),
    v30,
    v62,
    *(_QWORD *)(v60 + 56),
    *(_QWORD *)(v60 + 64));
  v31 = *(unsigned int **)(v13 + 48);
  v32 = &v31[4 * *(unsigned int *)(v13 + 56)];
  while ( v32 != v31 )
  {
    v33 = *((_QWORD *)v31 + 1);
    v34 = *v31;
    v31 += 4;
    sub_B99FD0(v30, v34, v33);
  }
  v35 = *(unsigned __int8 *)(a11 + 137) + *(unsigned __int8 *)(a11 + 136);
  v36 = v35 + 6;
  v37 = v35 - 1;
  if ( v37 < 0 )
    v37 = v36;
  if ( v56 != (*(_QWORD *)(a11 + 128) + (v37 >> 3)) / v55 )
    sub_91B8A0("a bitfield straddling elements of container type is not supported!", a2, 1);
  return v30;
}
