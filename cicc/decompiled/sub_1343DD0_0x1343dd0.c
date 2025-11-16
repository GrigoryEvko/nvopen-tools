// Function: sub_1343DD0
// Address: 0x1343dd0
//
int __fastcall sub_1343DD0(_BYTE *a1, __int64 a2, unsigned int *a3, __int64 a4, _QWORD *a5)
{
  unsigned __int64 v7; // r8
  __int64 v10; // r15
  unsigned __int64 v11; // r10
  unsigned __int64 v12; // rdx
  char v13; // al
  _BYTE *v14; // r11
  char v15; // r15
  bool v16; // zf
  __int64 v17; // r10
  size_t v18; // rdx
  unsigned __int64 v19; // r15
  _BYTE *v20; // r11
  unsigned __int64 v21; // rdi
  _BYTE *v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  _BYTE *v25; // rdi
  __int64 v26; // rax
  unsigned __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  unsigned __int64 v33; // [rsp+18h] [rbp-48h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+18h] [rbp-48h]
  size_t v36; // [rsp+18h] [rbp-48h]
  unsigned __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  unsigned __int64 v42; // [rsp+28h] [rbp-38h]
  size_t v43; // [rsp+28h] [rbp-38h]

  v7 = a5[2] & 0xFFFFFFFFFFFFF000LL;
  _InterlockedAdd64((volatile signed __int64 *)(*(_QWORD *)(a2 + 62224) + 64LL), v7);
  if ( *(_DWORD *)(a4 + 19424) != 1 )
    return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
  v10 = *((_QWORD *)a3 + 1);
  v11 = a5[1] & 0xFFFFFFFFFFFFF000LL;
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v10 == &off_49E8020 )
  {
    v15 = sub_1341100(a5[1] & 0xFFFFFFFFFFFFF000LL, 0, v7);
    goto LABEL_11;
  }
  if ( !*(_QWORD *)(v10 + 40) )
    goto LABEL_12;
  v12 = a5[2] & 0xFFFFFFFFFFFFF000LL;
  if ( !a1 )
  {
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v28 = a5[1] & 0xFFFFFFFFFFFFF000LL;
      v30 = a5[2] & 0xFFFFFFFFFFFFF000LL;
      v35 = v7;
      v39 = __readfsqword(0);
      v23 = sub_1313D30(v39 - 2664, 0);
      v7 = v35;
      v12 = v30;
      ++*(_BYTE *)(v23 + 1);
      v22 = (_BYTE *)v23;
      v11 = v28;
      if ( *(_BYTE *)(v23 + 816) )
      {
        v15 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v10 + 40))(
                v10,
                v28,
                v30,
                0,
                v35,
                *a3);
LABEL_25:
        v14 = (_BYTE *)(v39 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v14 = (_BYTE *)sub_1313D30(v39 - 2664, 0);
        goto LABEL_9;
      }
    }
    else
    {
      v21 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v39 = v21;
      v22 = (_BYTE *)(v21 - 2664);
    }
    v27 = v11;
    v29 = v12;
    v34 = v7;
    sub_1313A40(v22);
    v15 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v10 + 40))(
            v10,
            v27,
            v29,
            0,
            v34,
            *a3);
    goto LABEL_25;
  }
  ++a1[1];
  if ( a1[816] )
  {
    v13 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v10 + 40))(
            v10,
            v11,
            v12,
            0,
            v7,
            *a3);
  }
  else
  {
    v33 = v11;
    v38 = v12;
    v42 = v7;
    sub_1313A40(a1);
    v13 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v10 + 40))(
            v10,
            v33,
            v38,
            0,
            v42,
            *a3);
  }
  v14 = a1;
  v15 = v13;
LABEL_9:
  v16 = v14[1]-- == 1;
  if ( v16 )
    sub_1313A40(v14);
LABEL_11:
  if ( !v15 )
    return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
LABEL_12:
  v17 = *((_QWORD *)a3 + 1);
  v18 = a5[2] & 0xFFFFFFFFFFFFF000LL;
  v19 = a5[1] & 0xFFFFFFFFFFFFF000LL;
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v17 == &off_49E8020 )
  {
    sub_1341120(a5[1] & 0xFFFFFFFFFFFFF000LL, 0, v18);
    return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
  }
  if ( *(_QWORD *)(v17 + 48) )
  {
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v17 + 48))(
          v17,
          v19,
          v18,
          0,
          v18,
          *a3);
      }
      else
      {
        v41 = v17;
        v43 = v18;
        sub_1313A40(a1);
        (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v41 + 48))(
          v41,
          v19,
          v43,
          0,
          v43,
          *a3);
      }
      v20 = a1;
LABEL_18:
      v16 = v20[1]-- == 1;
      if ( v16 )
        sub_1313A40(v20);
      return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v32 = *((_QWORD *)a3 + 1);
      v37 = a5[2] & 0xFFFFFFFFFFFFF000LL;
      v40 = __readfsqword(0);
      v26 = sub_1313D30(v40 - 2664, 0);
      v18 = v37;
      v17 = v32;
      ++*(_BYTE *)(v26 + 1);
      v25 = (_BYTE *)v26;
      if ( *(_BYTE *)(v26 + 816) )
      {
        (*(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, _QWORD))(v32 + 48))(
          v32,
          v19,
          v37,
          0,
          v37,
          *a3);
LABEL_32:
        v20 = (_BYTE *)(v40 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v20 = (_BYTE *)sub_1313D30(v40 - 2664, 0);
        goto LABEL_18;
      }
    }
    else
    {
      v24 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v40 = v24;
      v25 = (_BYTE *)(v24 - 2664);
    }
    v31 = v17;
    v36 = v18;
    sub_1313A40(v25);
    (*(void (__fastcall **)(__int64, unsigned __int64, size_t, _QWORD, size_t, _QWORD))(v31 + 48))(
      v31,
      v19,
      v36,
      0,
      v36,
      *a3);
    goto LABEL_32;
  }
  return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a5);
}
