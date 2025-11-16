// Function: sub_1345440
// Address: 0x1345440
//
bool __fastcall sub_1345440(_BYTE *a1, unsigned int *a2, __int64 a3, __int64 a4, size_t a5)
{
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r13
  unsigned __int64 v8; // r14
  char v9; // r15
  _BYTE *v10; // r12
  unsigned __int64 v11; // r15
  __int64 v12; // r9
  unsigned __int64 v15; // r12
  _BYTE *v16; // rdi
  __int64 v17; // r9
  __int64 v18; // rax
  size_t v19; // [rsp+0h] [rbp-40h]
  size_t v20; // [rsp+0h] [rbp-40h]
  size_t v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a3 + 8);
  v6 = *(_QWORD *)(a3 + 16);
  v7 = *((_QWORD *)a2 + 1);
  v8 = v5 & 0xFFFFFFFFFFFFF000LL;
  if ( (__int64 (__fastcall **)(int, int, int, int, int, int, int))v7 != &off_49E8020 )
  {
    v9 = 1;
    if ( !*(_QWORD *)(v7 + 40) )
      return v9;
    v10 = a1;
    v11 = v6 & 0xFFFFFFFFFFFFF000LL;
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v12 = *a2;
      }
      else
      {
        v20 = a5;
        v23 = a4;
        sub_1313A40(a1);
        v12 = *a2;
        a5 = v20;
        a4 = v23;
      }
      v9 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, size_t, __int64))(v7 + 40))(
             v7,
             v8,
             v11,
             a4,
             a5,
             v12);
LABEL_7:
      if ( v10[1]-- == 1 )
        sub_1313A40(v10);
      return v9;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v15 = __readfsqword(0);
      v21 = a5;
      v24 = a4;
      v18 = sub_1313D30(v15 - 2664, 0);
      a4 = v24;
      a5 = v21;
      ++*(_BYTE *)(v18 + 1);
      v16 = (_BYTE *)v18;
      if ( *(_BYTE *)(v18 + 816) )
      {
        v17 = *a2;
LABEL_17:
        v9 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64, size_t, __int64))(v7 + 40))(
               v7,
               v8,
               v11,
               a4,
               a5,
               v17);
        v10 = (_BYTE *)(v15 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
          v10 = (_BYTE *)sub_1313D30((__int64)v10, 0);
        goto LABEL_7;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v15 = __readfsqword(0);
      v16 = (_BYTE *)(v15 - 2664);
    }
    v19 = a5;
    v22 = a4;
    sub_1313A40(v16);
    v17 = *a2;
    a5 = v19;
    a4 = v22;
    goto LABEL_17;
  }
  return sub_1341100(v8, a4, a5);
}
