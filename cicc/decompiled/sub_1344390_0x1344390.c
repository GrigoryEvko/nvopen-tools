// Function: sub_1344390
// Address: 0x1344390
//
unsigned __int64 *__fastcall sub_1344390(
        _BYTE *a1,
        __int64 a2,
        unsigned int *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        unsigned __int8 *a8)
{
  __int64 v12; // rsi
  unsigned __int64 *v13; // r15
  int *v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 (__fastcall **v16)(int, int, int, int, int, int, int); // r13
  __int64 v17; // rax
  _BYTE *v18; // r11
  __int64 v19; // r13
  unsigned __int64 v21; // r14
  signed __int64 v22; // rax
  unsigned __int64 v23; // r8
  signed __int64 v24; // r9
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int64 *v27; // rdx
  unsigned __int64 v29; // rax
  _BYTE *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  int *v33; // [rsp+8h] [rbp-58h]
  int *v34; // [rsp+8h] [rbp-58h]
  int *v35; // [rsp+8h] [rbp-58h]
  unsigned __int8 v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int8 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h]
  unsigned __int64 v42; // [rsp+18h] [rbp-48h]
  _BYTE v44[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v12 = *(_QWORD *)(a2 + 58392);
  v44[0] = a7;
  v13 = sub_1340A00(a1, v12);
  if ( !v13 )
    return v13;
  v14 = (int *)a3;
  v15 = (a6 + 4095) & 0xFFFFFFFFFFFFF000LL;
  v16 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))*((_QWORD *)a3 + 1);
  if ( v16 != &off_49E8020 )
  {
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v17 = (*v16)((int)v16, a4, a5, v15, (int)v44, (int)a8, *a3);
      }
      else
      {
        v34 = (int *)a3;
        v42 = v15;
        sub_1313A40(a1);
        v17 = (*v16)((int)v16, a4, a5, v42, (int)v44, (int)a8, *v34);
      }
      v18 = a1;
      v19 = v17;
LABEL_7:
      if ( v18[1]-- == 1 )
        sub_1313A40(v18);
      goto LABEL_9;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v35 = (int *)a3;
      v38 = v15;
      v41 = __readfsqword(0);
      v32 = sub_1313D30(v41 - 2664, 0);
      v15 = v38;
      v14 = v35;
      ++*(_BYTE *)(v32 + 1);
      v30 = (_BYTE *)v32;
      if ( *(_BYTE *)(v32 + 816) )
      {
        v31 = (*v16)((int)v16, a4, a5, v38, (int)v44, (int)a8, *v35);
LABEL_17:
        v19 = v31;
        v18 = (_BYTE *)(v41 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v18 = (_BYTE *)sub_1313D30(v41 - 2664, 0);
        goto LABEL_7;
      }
    }
    else
    {
      v29 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v41 = v29;
      v30 = (_BYTE *)(v29 - 2664);
    }
    v33 = v14;
    v37 = v15;
    sub_1313A40(v30);
    v31 = (*v16)((int)v16, a4, a5, v37, (int)v44, (int)a8, *v33);
    goto LABEL_17;
  }
  v19 = sub_1340EA0((__int64)a1, a4, a5, v15, (__int64)v44, (__int64)a8, *a3);
LABEL_9:
  if ( !v19 )
    goto LABEL_11;
  v21 = (unsigned __int64)v44[0] << 15;
  v36 = *a8;
  v40 = unk_4C6F2C8;
  v22 = sub_13441B0(a2);
  v23 = v13[2];
  v24 = v22;
  v25 = *v13;
  v26 = *(unsigned int *)(a2 + 19484);
  v13[1] = v19;
  v13[4] = v24;
  v13[2] = v23 & 0xFFF | a5;
  *v13 = ((unsigned __int64)v40 << 44)
       | ((unsigned __int64)v36 << 13) & 0xFFFFEFFFFFFFBFFFLL
       | v21 & 0xFFFFEFFFFFFFBFFFLL
       | v26 & 0xFFFFEFFFF0000FFFLL
       | v25 & 0xFFFFEFFFF0000000LL
       | 0xE800000;
  if ( (unsigned __int8)sub_1341BA0((__int64)a1, *(_QWORD *)(a2 + 58384), v13, 0xE8u, 0) )
  {
LABEL_11:
    v27 = v13;
    v13 = 0;
    sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), v27);
  }
  return v13;
}
