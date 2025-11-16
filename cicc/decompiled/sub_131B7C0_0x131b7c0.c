// Function: sub_131B7C0
// Address: 0x131b7c0
//
unsigned int *__fastcall sub_131B7C0(_BYTE *a1, unsigned int *a2, void *a3, size_t a4)
{
  __int64 v6; // r9
  _BYTE *v7; // r12
  char v8; // al
  _BYTE *v9; // r10
  char v10; // dl
  bool v11; // zf
  unsigned int *result; // rax
  __int64 v13; // r10
  char v14; // al
  _BYTE *v15; // r11
  char v16; // dl
  __int64 v17; // r10
  char v18; // al
  _BYTE *v19; // r11
  char v20; // dl
  __int64 v21; // r10
  unsigned __int64 v22; // rcx
  _BYTE *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  _BYTE *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  _BYTE *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  _BYTE *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+8h] [rbp-48h]
  unsigned __int64 v45; // [rsp+10h] [rbp-40h]
  unsigned __int64 v46; // [rsp+10h] [rbp-40h]
  unsigned __int64 v47; // [rsp+10h] [rbp-40h]
  unsigned __int64 v48; // [rsp+10h] [rbp-40h]
  __int64 v49; // [rsp+18h] [rbp-38h]
  char v50; // [rsp+18h] [rbp-38h]
  char v51; // [rsp+18h] [rbp-38h]
  char v52; // [rsp+18h] [rbp-38h]
  __int64 v53; // [rsp+18h] [rbp-38h]
  char v54; // [rsp+18h] [rbp-38h]
  __int64 v55; // [rsp+18h] [rbp-38h]
  char v56; // [rsp+18h] [rbp-38h]
  char v57; // [rsp+18h] [rbp-38h]
  __int64 v58; // [rsp+18h] [rbp-38h]

  if ( *((__int64 (__fastcall ***)(int, int, int, int, int, int, int))a2 + 1) != &off_49E8020 )
  {
    v6 = *((_QWORD *)a2 + 1);
    v7 = a1;
    if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v6 == &off_49E8020 )
    {
      v10 = sub_1341040(a3, a4);
      goto LABEL_10;
    }
    if ( !*(_QWORD *)(v6 + 8) )
      goto LABEL_14;
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v8 = (*(__int64 (__fastcall **)(__int64, void *, size_t, __int64, _QWORD))(v6 + 8))(v6, a3, a4, 1, *a2);
      }
      else
      {
        v49 = v6;
        sub_1313A40(a1);
        v8 = (*(__int64 (__fastcall **)(__int64, void *, size_t, __int64, _QWORD))(v49 + 8))(v49, a3, a4, 1, *a2);
      }
      v9 = a1;
      v10 = v8;
LABEL_8:
      v11 = v9[1]-- == 1;
      if ( v11 )
      {
        v50 = v10;
        sub_1313A40(v9);
        v10 = v50;
      }
LABEL_10:
      if ( !v10 )
        goto LABEL_11;
LABEL_14:
      v13 = *((_QWORD *)a2 + 1);
      if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v13 == &off_49E8020 )
      {
        v16 = sub_13410E0(a3, 0, a4);
        goto LABEL_22;
      }
      if ( !*(_QWORD *)(v13 + 32) )
        goto LABEL_23;
      if ( v7 )
      {
        ++v7[1];
        if ( v7[816] )
        {
          v14 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v13 + 32))(
                  v13,
                  a3,
                  a4,
                  0,
                  a4,
                  *a2);
        }
        else
        {
          v55 = v13;
          sub_1313A40(v7);
          v14 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v55 + 32))(
                  v55,
                  a3,
                  a4,
                  0,
                  a4,
                  *a2);
        }
        v15 = v7;
        v16 = v14;
LABEL_20:
        v11 = v15[1]-- == 1;
        if ( v11 )
        {
          v52 = v16;
          sub_1313A40(v15);
          v16 = v52;
        }
LABEL_22:
        if ( !v16 )
          goto LABEL_11;
LABEL_23:
        v17 = *((_QWORD *)a2 + 1);
        if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v17 == &off_49E8020 )
        {
          v20 = sub_1341120(a3, 0, a4);
          goto LABEL_31;
        }
        if ( !*(_QWORD *)(v17 + 48) )
          goto LABEL_32;
        if ( v7 )
        {
          ++v7[1];
          if ( v7[816] )
          {
            v18 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v17 + 48))(
                    v17,
                    a3,
                    a4,
                    0,
                    a4,
                    *a2);
          }
          else
          {
            v53 = v17;
            sub_1313A40(v7);
            v18 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v53 + 48))(
                    v53,
                    a3,
                    a4,
                    0,
                    a4,
                    *a2);
          }
          v19 = v7;
          v20 = v18;
LABEL_29:
          v11 = v19[1]-- == 1;
          if ( v11 )
          {
            v56 = v20;
            sub_1313A40(v19);
            v20 = v56;
          }
LABEL_31:
          if ( !v20 )
            goto LABEL_11;
LABEL_32:
          v21 = *((_QWORD *)a2 + 1);
          if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v21 == &off_49E8020 )
          {
            sub_1341100(a3, 0, a4);
            goto LABEL_11;
          }
          if ( !*(_QWORD *)(v21 + 40) )
            goto LABEL_11;
          if ( v7 )
          {
            ++v7[1];
            if ( v7[816] )
            {
              (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v21 + 40))(
                v21,
                a3,
                a4,
                0,
                a4,
                *a2);
            }
            else
            {
              v58 = v21;
              sub_1313A40(v7);
              (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v58 + 40))(
                v58,
                a3,
                a4,
                0,
                a4,
                *a2);
            }
LABEL_37:
            v11 = v7[1]-- == 1;
            if ( v11 )
              sub_1313A40(v7);
            goto LABEL_11;
          }
          if ( __readfsbyte(0xFFFFF8C8) )
          {
            v44 = *((_QWORD *)a2 + 1);
            v48 = __readfsqword(0);
            v36 = sub_1313D30(v48 - 2664, 0);
            v21 = v44;
            ++*(_BYTE *)(v36 + 1);
            v35 = (_BYTE *)v36;
            if ( *(_BYTE *)(v36 + 816) )
            {
              (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v44 + 40))(
                v44,
                a3,
                a4,
                0,
                a4,
                *a2);
LABEL_74:
              v7 = (_BYTE *)(v48 - 2664);
              if ( __readfsbyte(0xFFFFF8C8) )
                v7 = (_BYTE *)sub_1313D30(v48 - 2664, 0);
              goto LABEL_37;
            }
          }
          else
          {
            v34 = __readfsqword(0);
            __addfsbyte(0xFFFFF599, 1u);
            v48 = v34;
            v35 = (_BYTE *)(v34 - 2664);
          }
          v43 = v21;
          sub_1313A40(v35);
          (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v43 + 40))(v43, a3, a4, 0, a4, *a2);
          goto LABEL_74;
        }
        if ( __readfsbyte(0xFFFFF8C8) )
        {
          v42 = *((_QWORD *)a2 + 1);
          v47 = __readfsqword(0);
          v33 = sub_1313D30(v47 - 2664, 0);
          v17 = v42;
          ++*(_BYTE *)(v33 + 1);
          v31 = (_BYTE *)v33;
          if ( *(_BYTE *)(v33 + 816) )
          {
            v20 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v42 + 48))(
                    v42,
                    a3,
                    a4,
                    0,
                    a4,
                    *a2);
LABEL_66:
            v19 = (_BYTE *)(v47 - 2664);
            if ( __readfsbyte(0xFFFFF8C8) )
            {
              v57 = v20;
              v32 = sub_1313D30(v47 - 2664, 0);
              v20 = v57;
              v19 = (_BYTE *)v32;
            }
            goto LABEL_29;
          }
        }
        else
        {
          v30 = __readfsqword(0);
          __addfsbyte(0xFFFFF599, 1u);
          v47 = v30;
          v31 = (_BYTE *)(v30 - 2664);
        }
        v41 = v17;
        sub_1313A40(v31);
        v20 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v41 + 48))(
                v41,
                a3,
                a4,
                0,
                a4,
                *a2);
        goto LABEL_66;
      }
      if ( __readfsbyte(0xFFFFF8C8) )
      {
        v40 = *((_QWORD *)a2 + 1);
        v46 = __readfsqword(0);
        v29 = sub_1313D30(v46 - 2664, 0);
        v13 = v40;
        ++*(_BYTE *)(v29 + 1);
        v27 = (_BYTE *)v29;
        if ( *(_BYTE *)(v29 + 816) )
        {
          v16 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v40 + 32))(
                  v40,
                  a3,
                  a4,
                  0,
                  a4,
                  *a2);
LABEL_57:
          v15 = (_BYTE *)(v46 - 2664);
          if ( __readfsbyte(0xFFFFF8C8) )
          {
            v54 = v16;
            v28 = sub_1313D30(v46 - 2664, 0);
            v16 = v54;
            v15 = (_BYTE *)v28;
          }
          goto LABEL_20;
        }
      }
      else
      {
        v26 = __readfsqword(0);
        __addfsbyte(0xFFFFF599, 1u);
        v46 = v26;
        v27 = (_BYTE *)(v26 - 2664);
      }
      v39 = v13;
      sub_1313A40(v27);
      v16 = (*(__int64 (__fastcall **)(__int64, void *, size_t, _QWORD, size_t, _QWORD))(v39 + 32))(
              v39,
              a3,
              a4,
              0,
              a4,
              *a2);
      goto LABEL_57;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v38 = *((_QWORD *)a2 + 1);
      v45 = __readfsqword(0);
      v24 = sub_1313D30(v45 - 2664, 0);
      v6 = v38;
      ++*(_BYTE *)(v24 + 1);
      v23 = (_BYTE *)v24;
      if ( *(_BYTE *)(v24 + 816) )
      {
        v10 = (*(__int64 (__fastcall **)(__int64, void *, size_t, __int64, _QWORD))(v38 + 8))(v38, a3, a4, 1, *a2);
LABEL_51:
        v9 = (_BYTE *)(v45 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
        {
          v51 = v10;
          v25 = sub_1313D30(v45 - 2664, 0);
          v10 = v51;
          v9 = (_BYTE *)v25;
        }
        goto LABEL_8;
      }
    }
    else
    {
      v22 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v45 = v22;
      v23 = (_BYTE *)(v22 - 2664);
    }
    v37 = v6;
    sub_1313A40(v23);
    v10 = (*(__int64 (__fastcall **)(__int64, void *, size_t, __int64, _QWORD))(v37 + 8))(v37, a3, a4, 1, *a2);
    goto LABEL_51;
  }
  if ( (unsigned __int8)sub_1346920(a3, a4) && (unsigned __int8)sub_130CC10(a3, a4) && sub_130CD80(a3, a4) )
    sub_130CD50(a3, a4);
LABEL_11:
  result = &dword_4F96B94;
  if ( dword_4F96B94 )
  {
    result = (unsigned int *)unk_505F9C8;
    if ( !unk_505F9C8 )
      return (unsigned int *)sub_130CDC0(a3, a4);
  }
  return result;
}
