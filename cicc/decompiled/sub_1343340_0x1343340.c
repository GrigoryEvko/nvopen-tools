// Function: sub_1343340
// Address: 0x1343340
//
__int64 __fastcall sub_1343340(_BYTE *a1, __int64 a2, unsigned int *a3, unsigned __int64 *a4, _QWORD *a5)
{
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v13; // r10
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  char v17; // di
  unsigned __int64 v18; // r8
  unsigned int v19; // eax
  _BYTE *v20; // r11
  unsigned int v21; // r15d
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  _BYTE *v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rax
  unsigned int *v31; // [rsp+8h] [rbp-88h]
  unsigned int *v32; // [rsp+8h] [rbp-88h]
  unsigned int *v33; // [rsp+8h] [rbp-88h]
  unsigned __int64 v34; // [rsp+10h] [rbp-80h]
  unsigned __int64 v35; // [rsp+10h] [rbp-80h]
  unsigned __int64 v36; // [rsp+18h] [rbp-78h]
  unsigned __int64 v37; // [rsp+18h] [rbp-78h]
  unsigned __int64 v38; // [rsp+18h] [rbp-78h]
  unsigned __int64 v39; // [rsp+20h] [rbp-70h]
  unsigned __int64 v40; // [rsp+20h] [rbp-70h]
  unsigned __int64 v41; // [rsp+20h] [rbp-70h]
  unsigned __int64 v42; // [rsp+28h] [rbp-68h]
  unsigned __int64 v43; // [rsp+28h] [rbp-68h]
  unsigned __int64 v44; // [rsp+28h] [rbp-68h]
  unsigned __int64 v45; // [rsp+30h] [rbp-60h]
  unsigned __int64 v46; // [rsp+30h] [rbp-60h]
  unsigned int v47; // [rsp+38h] [rbp-58h]
  _BYTE v48[80]; // [rsp+40h] [rbp-50h] BYREF

  v10 = *a4;
  v11 = a4[2];
  v12 = *((_QWORD *)a3 + 1);
  v13 = a4[1] & 0xFFFFFFFFFFFFF000LL;
  v14 = a5[1] & 0xFFFFFFFFFFFFF000LL;
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v12 == &off_49E8020 )
  {
    v21 = sub_1341160((__int64)a1, v13, a5[1] & 0xFFFFFFFFFFFFF000LL);
    goto LABEL_9;
  }
  if ( !*(_QWORD *)(v12 + 64) )
    return 1;
  v15 = a5[2] & 0xFFFFFFFFFFFFF000LL;
  v16 = v11 & 0xFFFFFFFFFFFFF000LL;
  v17 = v10 >> 13;
  v18 = v15;
  v47 = v17 & 1;
  if ( !a1 )
  {
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v33 = a3;
      v35 = v13;
      v38 = v16;
      v41 = a5[1] & 0xFFFFFFFFFFFFF000LL;
      v44 = a5[2] & 0xFFFFFFFFFFFFF000LL;
      v45 = __readfsqword(0);
      v30 = sub_1313D30(v45 - 2664, 0);
      v18 = v44;
      v14 = v41;
      ++*(_BYTE *)(v30 + 1);
      v16 = v38;
      v28 = (_BYTE *)v30;
      v13 = v35;
      a3 = v33;
      if ( *(_BYTE *)(v30 + 816) )
      {
        v29 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v12 + 64))(
                v12,
                v35,
                v38,
                v41,
                v44,
                v47,
                *v33);
LABEL_24:
        v20 = (_BYTE *)(v45 - 2664);
        v21 = v29;
        if ( __readfsbyte(0xFFFFF8C8) )
          v20 = (_BYTE *)sub_1313D30(v45 - 2664, 0);
        goto LABEL_7;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v45 = __readfsqword(0);
      v28 = (_BYTE *)(v45 - 2664);
    }
    v31 = a3;
    v34 = v13;
    v36 = v16;
    v39 = v14;
    v42 = v18;
    sub_1313A40(v28);
    v29 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v12 + 64))(
            v12,
            v34,
            v36,
            v39,
            v42,
            v47,
            *v31);
    goto LABEL_24;
  }
  ++a1[1];
  if ( a1[816] )
  {
    v19 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v12 + 64))(
            v12,
            v13,
            v16,
            v14,
            v15,
            v17 & 1,
            *a3);
  }
  else
  {
    v32 = a3;
    v37 = v13;
    v40 = v16;
    v43 = v14;
    v46 = v15;
    sub_1313A40(a1);
    v19 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v12 + 64))(
            v12,
            v37,
            v40,
            v43,
            v46,
            v17 & 1,
            *v32);
  }
  v20 = a1;
  v21 = v19;
LABEL_7:
  if ( v20[1]-- == 1 )
    sub_1313A40(v20);
LABEL_9:
  if ( (_BYTE)v21 )
    return 1;
  sub_1342660((__int64)a1, *(_QWORD *)(a2 + 58384), (__int64)v48, (__int64)a4, (__int64)a5);
  v23 = *a4;
  v24 = a4[2];
  *a4 &= 0xFFFFFFFFFFF1FFFFLL;
  a4[2] = v24 & 0xFFF | ((a5[2] & 0xFFFFFFFFFFFFF000LL) + (v24 & 0xFFFFFFFFFFFFF000LL));
  v25 = a5[4];
  if ( a4[4] <= v25 )
    v25 = a4[4];
  a4[4] = v25;
  v26 = (unsigned __int16)v23 & 0x8000;
  if ( (v23 & 0x8000) != 0 )
    v26 = *a5 & 0x8000LL;
  *a4 = v26 | v23 & 0xFFFFFFFFFFF17FFFLL;
  sub_1342700((__int64)a1, *(_QWORD *)(a2 + 58384), (__int64)v48, a4);
  sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
  return v21;
}
