// Function: sub_13448D0
// Address: 0x13448d0
//
__int64 __fastcall sub_13448D0(_BYTE *a1, unsigned int *a2, __int64 *a3, __int64 a4, size_t a5)
{
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r14
  unsigned __int64 v9; // r15
  _BYTE *v10; // r12
  unsigned __int64 v11; // rdx
  unsigned int v12; // r13d
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v17; // r12
  _BYTE *v18; // rdi
  __int64 v19; // r9
  __int64 v20; // rax
  size_t v21; // [rsp+8h] [rbp-48h]
  size_t v22; // [rsp+8h] [rbp-48h]
  size_t v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  unsigned __int64 v27; // [rsp+18h] [rbp-38h]
  unsigned __int64 v28; // [rsp+18h] [rbp-38h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]

  v6 = a3[1];
  v7 = a3[2];
  v8 = *((_QWORD *)a2 + 1);
  v9 = v6 & 0xFFFFFFFFFFFFF000LL;
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v8 != &off_49E8020 )
  {
    if ( !*(_QWORD *)(v8 + 32) )
    {
      v14 = *a3;
      v12 = 1;
      v15 = *a3 & 0x2000;
      goto LABEL_10;
    }
    v10 = a1;
    v11 = v7 & 0xFFFFFFFFFFFFF000LL;
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, size_t, _QWORD))(v8 + 32))(
                v8,
                v9,
                v11,
                a4,
                a5,
                *a2);
      }
      else
      {
        v22 = a5;
        v25 = a4;
        v28 = v11;
        sub_1313A40(a1);
        v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, size_t, _QWORD))(v8 + 32))(
                v8,
                v9,
                v28,
                v25,
                v22,
                *a2);
      }
LABEL_6:
      if ( v10[1]-- == 1 )
        sub_1313A40(v10);
      goto LABEL_8;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v17 = __readfsqword(0);
      v23 = a5;
      v26 = a4;
      v29 = v11;
      v20 = sub_1313D30(v17 - 2664, 0);
      v11 = v29;
      a4 = v26;
      ++*(_BYTE *)(v20 + 1);
      v18 = (_BYTE *)v20;
      a5 = v23;
      if ( *(_BYTE *)(v20 + 816) )
      {
        v19 = *a2;
LABEL_19:
        v12 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, size_t, __int64))(v8 + 32))(
                v8,
                v9,
                v11,
                a4,
                a5,
                v19);
        v10 = (_BYTE *)(v17 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v10 = (_BYTE *)sub_1313D30((__int64)v10, 0);
        goto LABEL_6;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v17 = __readfsqword(0);
      v18 = (_BYTE *)(v17 - 2664);
    }
    v21 = a5;
    v24 = a4;
    v27 = v11;
    sub_1313A40(v18);
    v19 = *a2;
    a5 = v21;
    a4 = v24;
    v11 = v27;
    goto LABEL_19;
  }
  v12 = sub_13410E0(v9, a4, a5);
LABEL_8:
  v14 = *a3;
  v15 = *a3 & 0x2000;
  if ( (*a3 & 0x2000) != 0 )
    v15 = (unsigned __int64)((_BYTE)v12 != 0) << 13;
LABEL_10:
  BYTE1(v14) &= ~0x20u;
  *a3 = v15 | v14;
  return v12;
}
