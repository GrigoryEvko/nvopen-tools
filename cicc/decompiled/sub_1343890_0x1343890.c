// Function: sub_1343890
// Address: 0x1343890
//
unsigned __int64 *__fastcall sub_1343890(
        _BYTE *a1,
        __int64 a2,
        unsigned int *a3,
        unsigned __int64 *a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdx
  __int64 v13; // r9
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r8
  unsigned __int64 v19; // r9
  unsigned __int64 v20; // rcx
  __int64 v21; // r11
  __int64 v22; // r9
  __int64 v23; // rdx
  char v24; // al
  _BYTE *v25; // r10
  char v26; // dl
  _BYTE *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // [rsp+4h] [rbp-8Ch]
  unsigned int v33; // [rsp+4h] [rbp-8Ch]
  unsigned int v34; // [rsp+4h] [rbp-8Ch]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  unsigned __int64 v39; // [rsp+18h] [rbp-78h]
  unsigned __int64 v40; // [rsp+20h] [rbp-70h]
  char v42; // [rsp+30h] [rbp-60h]
  char v43; // [rsp+30h] [rbp-60h]
  __int64 *v45[10]; // [rsp+40h] [rbp-50h] BYREF

  if ( !*(_QWORD *)(*((_QWORD *)a3 + 1) + 56LL) )
    return 0;
  v10 = sub_1340A00(a1, *(_QWORD *)(a2 + 58392));
  v11 = v10;
  if ( !v10 )
    return 0;
  v12 = *a4;
  v13 = (__int64)v10;
  v14 = a5 + (a4[1] & 0xFFFFFFFFFFFFF000LL);
  v15 = *a4 & 0xFFF;
  v16 = *v10 & 0xFFFFFFFFF001E000LL;
  v11[4] = a4[4];
  v11[1] = v14;
  v11[2] = a6 | v11[2] & 0xFFF;
  *v11 = v12 & 0x2000 | (unsigned __int16)v12 & 0x8000 | v12 & 0xE0000 | (v15 | v16) & 0xFFFFEFFFF00E0FFFLL | 0xE800000;
  v17 = *(_QWORD *)(a2 + 58384);
  if ( sub_13424E0((__int64)a1, v17, (__int64)v45, (__int64)a4, a5, v13) )
    goto LABEL_14;
  v19 = *a4;
  v20 = a4[1];
  v21 = *((_QWORD *)a3 + 1);
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v21 == &off_49E8020 )
  {
    v26 = sub_1341140(a1, v17, a6, v20, v18, v19);
    goto LABEL_12;
  }
  if ( !*(_QWORD *)(v21 + 56) )
  {
LABEL_14:
    sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), v11);
    return 0;
  }
  v40 = v20 & 0xFFFFFFFFFFFFF000LL;
  v22 = (v19 >> 13) & 1;
  v23 = a5 + a6;
  if ( !a1 )
  {
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v34 = v22;
      v36 = *((_QWORD *)a3 + 1);
      v39 = __readfsqword(0);
      v31 = sub_1313D30(v39 - 2664, 0);
      v23 = a5 + a6;
      v21 = v36;
      ++*(_BYTE *)(v31 + 1);
      v29 = (_BYTE *)v31;
      LODWORD(v22) = v34;
      if ( *(_BYTE *)(v31 + 816) )
      {
        v26 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, _QWORD, _QWORD))(v36 + 56))(
                v36,
                v40,
                v23,
                a5,
                a6,
                v34,
                *a3);
LABEL_20:
        v25 = (_BYTE *)(v39 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
        {
          v42 = v26;
          v30 = sub_1313D30(v39 - 2664, 0);
          v26 = v42;
          v25 = (_BYTE *)v30;
        }
        goto LABEL_10;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v39 = __readfsqword(0);
      v29 = (_BYTE *)(v39 - 2664);
    }
    v32 = v22;
    v35 = v21;
    v37 = v23;
    sub_1313A40(v29);
    v26 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, _QWORD, _QWORD))(v35 + 56))(
            v35,
            v40,
            v37,
            a5,
            a6,
            v32,
            *a3);
    goto LABEL_20;
  }
  ++a1[1];
  if ( a1[816] )
  {
    v24 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, __int64, _QWORD))(v21 + 56))(
            v21,
            v40,
            v23,
            a5,
            a6,
            v22,
            *a3);
  }
  else
  {
    v33 = v22;
    v38 = v21;
    sub_1313A40(a1);
    v24 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, _QWORD, _QWORD))(v38 + 56))(
            v38,
            v40,
            a5 + a6,
            a5,
            a6,
            v33,
            *a3);
  }
  v25 = a1;
  v26 = v24;
LABEL_10:
  if ( v25[1]-- == 1 )
  {
    v43 = v26;
    sub_1313A40(v25);
    v26 = v43;
  }
LABEL_12:
  if ( v26 )
    goto LABEL_14;
  a4[2] = a5 | a4[2] & 0xFFF;
  sub_1342610((__int64)a1, *(_QWORD *)(a2 + 58384), v45, a4, a5, v11);
  return v11;
}
