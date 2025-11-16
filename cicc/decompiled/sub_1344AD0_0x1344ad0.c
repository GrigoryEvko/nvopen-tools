// Function: sub_1344AD0
// Address: 0x1344ad0
//
int __fastcall sub_1344AD0(_BYTE *a1, __int64 a2, unsigned int *a3, unsigned __int64 *a4)
{
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rcx
  char v12; // al
  _BYTE *v13; // r11
  char v14; // bl
  bool v15; // zf
  unsigned __int64 v17; // rax
  __int64 v18; // rbx
  size_t v19; // rdx
  __int64 v20; // rbx
  char v21; // al
  _BYTE *v22; // r11
  char v23; // bl
  unsigned __int64 v24; // rax
  _BYTE *v25; // rdi
  size_t v26; // rdx
  _BYTE *v27; // r11
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  _BYTE *v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  _BYTE *v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+8h] [rbp-58h]
  unsigned int v37; // [rsp+10h] [rbp-50h]
  unsigned int v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+10h] [rbp-50h]
  unsigned __int64 v40; // [rsp+10h] [rbp-50h]
  size_t v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43; // [rsp+18h] [rbp-48h]
  unsigned __int64 v44; // [rsp+18h] [rbp-48h]
  size_t v45; // [rsp+18h] [rbp-48h]
  unsigned __int64 v46; // [rsp+20h] [rbp-40h]
  unsigned __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+20h] [rbp-40h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  unsigned __int64 v50; // [rsp+28h] [rbp-38h]
  unsigned __int64 v51; // [rsp+28h] [rbp-38h]
  unsigned __int64 v52; // [rsp+28h] [rbp-38h]
  size_t v53; // [rsp+28h] [rbp-38h]
  unsigned __int64 v54; // [rsp+28h] [rbp-38h]
  size_t v55; // [rsp+28h] [rbp-38h]

  if ( *((__int64 (__fastcall ***)(int, int, int, int, int, int, int))a3 + 1) != &off_49E8020 )
  {
    if ( !*(_QWORD *)(*((_QWORD *)a3 + 1) + 8LL) )
      goto LABEL_15;
    if ( (*a4 & 0x10000) == 0 )
    {
LABEL_4:
      sub_1341E90((__int64)a1, *(_QWORD *)(a2 + 58384), (__int64)a4);
      v7 = *a4;
      v8 = a4[1] & 0xFFFFFFFFFFFFF000LL;
      v9 = a4[2] & 0xFFFFFFFFFFFFF000LL;
      a4[1] = v8;
      v10 = *((_QWORD *)a3 + 1);
      if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v10 == &off_49E8020 )
      {
        v14 = sub_1341040(v8, v9);
        goto LABEL_12;
      }
      if ( !*(_QWORD *)(v10 + 8) )
      {
LABEL_14:
        sub_1341BA0((__int64)a1, *(_QWORD *)(a2 + 58384), a4, 0xE8u, 0);
        goto LABEL_15;
      }
      v11 = (v7 >> 13) & 1;
      if ( a1 )
      {
        ++a1[1];
        if ( a1[816] )
        {
          v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, _QWORD))(v10 + 8))(
                  v10,
                  v8,
                  v9,
                  v11,
                  *a3);
        }
        else
        {
          v38 = v11;
          v48 = v10;
          v51 = v9;
          sub_1313A40(a1);
          v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v48 + 8))(
                  v48,
                  v8,
                  v51,
                  v38,
                  *a3);
        }
        v13 = a1;
        v14 = v12;
LABEL_10:
        v15 = v13[1]-- == 1;
        if ( v15 )
          sub_1313A40(v13);
LABEL_12:
        if ( !v14 )
          return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a4);
        goto LABEL_14;
      }
      if ( __readfsbyte(0xFFFFF8C8) )
      {
        v35 = v11;
        v39 = *((_QWORD *)a3 + 1);
        v44 = v9;
        v50 = __readfsqword(0);
        v28 = sub_1313D30(v50 - 2664, 0);
        v9 = v44;
        ++*(_BYTE *)(v28 + 1);
        v25 = (_BYTE *)v28;
        v10 = v39;
        LODWORD(v11) = v35;
        if ( *(_BYTE *)(v28 + 816) )
        {
          v14 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v39 + 8))(
                  v39,
                  v8,
                  v44,
                  v35,
                  *a3);
LABEL_35:
          v13 = (_BYTE *)(v50 - 2664);
          if ( __readfsbyte(0xFFFFF8C8) )
            v13 = (_BYTE *)sub_1313D30(v50 - 2664, 0);
          goto LABEL_10;
        }
      }
      else
      {
        v24 = __readfsqword(0);
        __addfsbyte(0xFFFFF599, 1u);
        v50 = v24;
        v25 = (_BYTE *)(v24 - 2664);
      }
      v37 = v11;
      v42 = v10;
      v47 = v9;
      sub_1313A40(v25);
      v14 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, _QWORD))(v42 + 8))(
              v42,
              v8,
              v47,
              v37,
              *a3);
      goto LABEL_35;
    }
LABEL_21:
    sub_130D990((__int64)a1, (__int64)a3, a4, *(_QWORD *)(a2 + 58384), 1, 1);
    goto LABEL_4;
  }
  if ( !unk_4C6F2C8 )
  {
    if ( (*a4 & 0x10000) == 0 )
      goto LABEL_4;
    goto LABEL_21;
  }
LABEL_15:
  v17 = *a4;
  v18 = 0x8000;
  if ( (*a4 & 0x2000) != 0 )
  {
    if ( !(unsigned __int8)sub_13448D0(a1, a3, (__int64 *)a4, 0, a4[2] & 0xFFFFFFFFFFFFF000LL) )
      goto LABEL_18;
    v19 = a4[2] & 0xFFFFFFFFFFFFF000LL;
    v46 = a4[1] & 0xFFFFFFFFFFFFF000LL;
    v20 = *((_QWORD *)a3 + 1);
    if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v20 == &off_49E8020 )
    {
      v23 = sub_1341120(v46, 0, v19);
      goto LABEL_30;
    }
    if ( !*(_QWORD *)(v20 + 48) )
      goto LABEL_38;
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v21 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v20 + 48))(
                v20,
                v46,
                v19,
                0,
                v19,
                *a3);
      }
      else
      {
        v53 = v19;
        sub_1313A40(a1);
        v21 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v20 + 48))(
                v20,
                v46,
                v53,
                0,
                v53,
                *a3);
      }
      v22 = a1;
      v23 = v21;
LABEL_28:
      v15 = v22[1]-- == 1;
      if ( v15 )
        sub_1313A40(v22);
LABEL_30:
      if ( !v23 )
      {
        v17 = *a4;
        v18 = 0x8000;
        goto LABEL_16;
      }
LABEL_38:
      v17 = *a4;
      v18 = 0;
      if ( ((*a4 >> 17) & 7) == 2 )
        goto LABEL_16;
      v26 = a4[2] & 0xFFFFFFFFFFFFF000LL;
      v43 = a4[1] & 0xFFFFFFFFFFFFF000LL;
      v49 = *((_QWORD *)a3 + 1);
      if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v49 == &off_49E8020 )
      {
        sub_1341100(v43, 0, v26);
        goto LABEL_18;
      }
      if ( !*(_QWORD *)(v49 + 40) )
      {
LABEL_18:
        v17 = *a4;
        goto LABEL_16;
      }
      if ( a1 )
      {
        ++a1[1];
        if ( a1[816] )
        {
          (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v49 + 40))(
            v49,
            v43,
            v26,
            0,
            v26,
            *a3);
        }
        else
        {
          v55 = v26;
          sub_1313A40(a1);
          (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v49 + 40))(
            v49,
            v43,
            v55,
            0,
            v55,
            *a3);
        }
        v27 = a1;
LABEL_45:
        v15 = v27[1]-- == 1;
        if ( v15 )
          sub_1313A40(v27);
        v17 = *a4;
        v18 = 0;
        goto LABEL_16;
      }
      if ( __readfsbyte(0xFFFFF8C8) )
      {
        v36 = a4[2] & 0xFFFFFFFFFFFFF000LL;
        v54 = __readfsqword(0);
        v34 = sub_1313D30(v54 - 2664, 0);
        v26 = v36;
        ++*(_BYTE *)(v34 + 1);
        v33 = (_BYTE *)v34;
        if ( *(_BYTE *)(v34 + 816) )
        {
          (*(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v49 + 40))(
            v49,
            v43,
            v36,
            0,
            v36,
            *a3);
LABEL_63:
          v27 = (_BYTE *)(v54 - 2664);
          if ( __readfsbyte(0xFFFFF8C8) )
            v27 = (_BYTE *)sub_1313D30(v54 - 2664, 0);
          goto LABEL_45;
        }
      }
      else
      {
        v32 = __readfsqword(0);
        __addfsbyte(0xFFFFF599, 1u);
        v54 = v32;
        v33 = (_BYTE *)(v32 - 2664);
      }
      v41 = v26;
      sub_1313A40(v33);
      (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v49 + 40))(
        v49,
        v43,
        v41,
        0,
        v41,
        *a3);
      goto LABEL_63;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v40 = a4[2] & 0xFFFFFFFFFFFFF000LL;
      v52 = __readfsqword(0);
      v31 = sub_1313D30(v52 - 2664, 0);
      v19 = v40;
      ++*(_BYTE *)(v31 + 1);
      v30 = (_BYTE *)v31;
      if ( *(_BYTE *)(v31 + 816) )
      {
        v23 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v20 + 48))(
                v20,
                v46,
                v40,
                0,
                v40,
                *a3);
LABEL_54:
        v22 = (_BYTE *)(v52 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v22 = (_BYTE *)sub_1313D30(v52 - 2664, 0);
        goto LABEL_28;
      }
    }
    else
    {
      v29 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v52 = v29;
      v30 = (_BYTE *)(v29 - 2664);
    }
    v45 = v19;
    sub_1313A40(v30);
    v23 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v20 + 48))(
            v20,
            v46,
            v45,
            0,
            v45,
            *a3);
    goto LABEL_54;
  }
LABEL_16:
  BYTE1(v17) &= ~0x80u;
  *a4 = v18 | v17;
  return sub_13451C0(a1, a2, a3, a2 + 38936, a4);
}
