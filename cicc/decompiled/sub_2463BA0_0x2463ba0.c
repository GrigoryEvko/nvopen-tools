// Function: sub_2463BA0
// Address: 0x2463ba0
//
__int64 __fastcall sub_2463BA0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // r13
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int8 *v10; // r10
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v12; // rax
  _BYTE *v13; // r15
  __int64 v14; // rax
  unsigned int **v15; // rdi
  __int64 v16; // r14
  unsigned int *v18; // rax
  unsigned int v19; // eax
  unsigned int v20; // r14d
  __int64 v21; // r15
  __int64 v22; // rdi
  unsigned int **v23; // r14
  _BYTE *v24; // rax
  __int64 v25; // rax
  __int64 v26; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-C8h]
  unsigned int *v28; // [rsp+18h] [rbp-C8h]
  __int64 v29; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v30; // [rsp+18h] [rbp-C8h]
  char v31[32]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v32; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v33; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v34; // [rsp+58h] [rbp-88h]
  __int16 v35; // [rsp+70h] [rbp-70h]
  _BYTE v36[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v37; // [rsp+A0h] [rbp-40h]

  v3 = (_BYTE *)a2;
  if ( !**(_BYTE **)a1 )
    goto LABEL_2;
  v29 = *(_QWORD *)(a2 + 8);
  v19 = sub_BCB060(v29);
  v20 = v19 - 1;
  v34 = v19;
  v21 = 1LL << ((unsigned __int8)v19 - 1);
  if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v33, 0, 0);
    if ( v34 > 0x40 )
    {
      *(_QWORD *)(v33 + 8LL * (v20 >> 6)) |= v21;
      v22 = *(_QWORD *)(a2 + 8);
      goto LABEL_16;
    }
    v22 = *(_QWORD *)(a2 + 8);
  }
  else
  {
    v33 = 0;
    v22 = v29;
  }
  v33 |= v21;
LABEL_16:
  v23 = *(unsigned int ***)(a1 + 8);
  v37 = 257;
  v24 = (_BYTE *)sub_AD8D80(v22, (__int64)&v33);
  v3 = (_BYTE *)sub_A825B0(v23, (_BYTE *)a2, v24, (__int64)v36);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
LABEL_2:
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(a3 + 8);
  v35 = 257;
  v32 = 257;
  v8 = sub_AD62B0(v7);
  v9 = *(_QWORD *)(v6 + 80);
  v10 = (unsigned __int8 *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v9 + 16LL);
  if ( v11 != sub_9202E0 )
  {
    v30 = v10;
    v25 = v11(v9, 30u, (_BYTE *)a3, v10);
    v10 = v30;
    v13 = (_BYTE *)v25;
    goto LABEL_8;
  }
  if ( *(_BYTE *)a3 <= 0x15u && *v10 <= 0x15u )
  {
    v27 = v10;
    if ( (unsigned __int8)sub_AC47B0(30) )
      v12 = sub_AD5570(30, a3, v27, 0, 0);
    else
      v12 = sub_AABE40(0x1Eu, (unsigned __int8 *)a3, v27);
    v10 = v27;
    v13 = (_BYTE *)v12;
LABEL_8:
    if ( v13 )
      goto LABEL_9;
  }
  v37 = 257;
  v13 = (_BYTE *)sub_B504D0(30, a3, (__int64)v10, (__int64)v36, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, char *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 88) + 16LL))(
    *(_QWORD *)(v6 + 88),
    v13,
    v31,
    *(_QWORD *)(v6 + 56),
    *(_QWORD *)(v6 + 64));
  v18 = *(unsigned int **)v6;
  v26 = *(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8);
  if ( *(_QWORD *)v6 != v26 )
  {
    do
    {
      v28 = v18;
      sub_B99FD0((__int64)v13, *v18, *((_QWORD *)v18 + 1));
      v18 = v28 + 4;
    }
    while ( (unsigned int *)v26 != v28 + 4 );
  }
LABEL_9:
  v14 = sub_A82350((unsigned int **)v6, v3, v13, (__int64)&v33);
  v15 = *(unsigned int ***)(a1 + 8);
  v16 = v14;
  v37 = 257;
  sub_A82480(v15, v3, (_BYTE *)a3, (__int64)v36);
  return v16;
}
