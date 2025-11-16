// Function: sub_8BF710
// Address: 0x8bf710
//
__int64 __fastcall sub_8BF710(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v3; // r15d
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // rdi
  __int64 v13; // [rsp+10h] [rbp-40h]

  v3 = dword_4D04734;
  v13 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  sub_854AB0();
  a2 = (unsigned int)a2;
  v6 = (__int64)a3;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  v7 = *(unsigned __int8 *)(v13 + 4);
  v8 = (unsigned int)(v7 - 3);
  if ( (unsigned __int8)(v7 - 3) <= 1u || !(_BYTE)v7 )
    goto LABEL_3;
  if ( (_BYTE)v7 != 6 || !unk_4D0473C )
  {
    a2 = 758;
    a1 = 8;
    sub_684AC0(8u, 0x2F6u);
LABEL_14:
    sub_7BE180(a1, a2, v8, v6, v4, v5);
    goto LABEL_10;
  }
  if ( dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
    goto LABEL_14;
LABEL_3:
  dword_4D04734 = 0;
  v9 = 16;
  if ( (a2 & 1) == 0 )
    v9 = (a2 & 2) == 0 ? 15 : 18;
  if ( dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  sub_8BE350(a1, (unsigned int)v9, 0, a3, v9, v5);
  if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
    sub_87DD20(dword_4F04C40);
  if ( dword_4F077C4 == 2 )
  {
    v11 = (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) &= ~8u;
    if ( *(_QWORD *)(qword_4F04C68[0] + 776 * v11 + 456) )
      sub_8845B0(v11);
  }
LABEL_10:
  result = qword_4F061C8;
  dword_4D04734 = v3;
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  return result;
}
