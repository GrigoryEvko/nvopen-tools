// Function: sub_86F430
// Address: 0x86f430
//
__int64 __fastcall sub_86F430(_BYTE *a1)
{
  __int64 v1; // r12
  __int64 v2; // r14
  int v3; // edx
  char v4; // r13
  bool v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r9
  __int64 result; // rax
  _QWORD *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax

  v1 = (__int64)a1;
  v2 = *((_QWORD *)a1 + 10);
  v3 = qword_4F5FD78;
  v4 = *(_BYTE *)(v2 + 24);
  *(_BYTE *)(v2 + 24) = qword_4F5FD78 & 1 | v4 & 0xFE;
  v5 = (v4 & 2) != 0;
  v6 = dword_4D047EC;
  if ( dword_4D047EC )
  {
    if ( unk_4D047E8 )
    {
      if ( v3 )
      {
        a1 = (_BYTE *)qword_4F5FD68;
        if ( *(_BYTE *)(qword_4F5FD68 + 32) )
        {
          v6 = *(_QWORD *)(qword_4F5FD68 + 16);
          a1 = sub_86B560(qword_4F5FD68, v6);
          if ( a1 )
          {
            v6 = (unsigned int)qword_4F5FD78;
            sub_86B010((__int64)a1, qword_4F5FD78);
          }
        }
      }
    }
  }
  sub_86F030();
  if ( *(_QWORD *)(*(_QWORD *)(v1 + 80) + 16LL) )
  {
    sub_733F40();
LABEL_6:
    result = unk_4D03B90;
    if ( unk_4D03B90 >= 0 && !v5 )
      return sub_86C020(v1);
    return result;
  }
  v12 = (_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
  if ( !v5 )
  {
    v16 = v12[23];
    if ( v16 )
    {
      *(_QWORD *)(v2 + 8) = v16;
      *(_QWORD *)(v16 + 80) = v1;
    }
    sub_863FC0((__int64)a1, v6, v7, v8, v9, v10);
    goto LABEL_6;
  }
  v13 = v12[61];
  v14 = v12[62];
  if ( v13 && (*(_BYTE *)(v13 + 1) & 4) != 0 )
  {
    a1 = (_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
    sub_732EF0((__int64)a1);
  }
  v15 = v12[23];
  if ( v15 )
  {
    *(_QWORD *)(v2 + 8) = v15;
    *(_QWORD *)(v15 + 80) = v1;
  }
  result = (__int64)sub_863FC0((__int64)a1, v6, v7, v8, v9, v10);
  if ( v14 )
  {
    qword_4F06BC0 = v14;
    return (__int64)&qword_4F06BC0;
  }
  return result;
}
