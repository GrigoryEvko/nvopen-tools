// Function: sub_13446B0
// Address: 0x13446b0
//
int __fastcall sub_13446B0(_BYTE *a1, __int64 a2, unsigned int *a3, unsigned __int64 *a4)
{
  unsigned __int64 v7; // rcx
  void *v8; // r14
  size_t v9; // rdx
  __int64 v10; // r10
  __int64 v11; // rcx
  _BYTE *v12; // r9
  unsigned __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // [rsp+4h] [rbp-4Ch]
  unsigned int v19; // [rsp+4h] [rbp-4Ch]
  unsigned int v20; // [rsp+4h] [rbp-4Ch]
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  size_t v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  size_t v25; // [rsp+10h] [rbp-40h]
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]
  size_t v27; // [rsp+18h] [rbp-38h]

  v7 = *a4;
  if ( (v7 & 0x10000) != 0 )
  {
    sub_130DA90((__int64)a1, (__int64)a3, a4);
    v7 = *a4;
  }
  v8 = (void *)(a4[1] & 0xFFFFFFFFFFFFF000LL);
  v9 = a4[2] & 0xFFFFFFFFFFFFF000LL;
  a4[1] = (unsigned __int64)v8;
  v10 = *((_QWORD *)a3 + 1);
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v10 == &off_49E8020 )
  {
    sub_1341080(v8, v9);
    return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a4);
  }
  if ( *(_QWORD *)(v10 + 16) )
  {
    v11 = (v7 >> 13) & 1;
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        (*(void (__fastcall **)(__int64, void *, size_t, __int64, _QWORD))(v10 + 16))(v10, v8, v9, v11, *a3);
      }
      else
      {
        v19 = v11;
        v24 = v10;
        v27 = v9;
        sub_1313A40(a1);
        (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, _QWORD))(v24 + 16))(v24, v8, v27, v19, *a3);
      }
      v12 = a1;
LABEL_9:
      if ( v12[1]-- == 1 )
        sub_1313A40(v12);
      return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a4);
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v20 = v11;
      v22 = *((_QWORD *)a3 + 1);
      v25 = v9;
      v26 = __readfsqword(0);
      v17 = sub_1313D30(v26 - 2664, 0);
      v9 = v25;
      v10 = v22;
      ++*(_BYTE *)(v17 + 1);
      v16 = (_BYTE *)v17;
      LODWORD(v11) = v20;
      if ( *(_BYTE *)(v17 + 816) )
      {
        (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, _QWORD))(v22 + 16))(v22, v8, v25, v20, *a3);
LABEL_19:
        v12 = (_BYTE *)(v26 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v12 = (_BYTE *)sub_1313D30(v26 - 2664, 0);
        goto LABEL_9;
      }
    }
    else
    {
      v15 = __readfsqword(0);
      __addfsbyte(0xFFFFF599, 1u);
      v26 = v15;
      v16 = (_BYTE *)(v15 - 2664);
    }
    v18 = v11;
    v21 = v10;
    v23 = v9;
    sub_1313A40(v16);
    (*(void (__fastcall **)(__int64, void *, size_t, _QWORD, _QWORD))(v21 + 16))(v21, v8, v23, v18, *a3);
    goto LABEL_19;
  }
  return sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), a4);
}
