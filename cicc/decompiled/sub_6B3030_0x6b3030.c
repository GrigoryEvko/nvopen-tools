// Function: sub_6B3030
// Address: 0x6b3030
//
__int64 __fastcall sub_6B3030(__int64 *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned __int16 v6; // bx
  int v7; // r8d
  __int64 result; // rax
  char v9; // al
  __int64 v10; // rax
  char i; // dl
  int v12; // [rsp+4h] [rbp-2FCh] BYREF
  __int64 v13; // [rsp+8h] [rbp-2F8h] BYREF
  _BYTE v14[352]; // [rsp+10h] [rbp-2F0h] BYREF
  __int64 v15[50]; // [rsp+170h] [rbp-190h] BYREF

  if ( dword_4F077C0 && *(_BYTE *)(qword_4D03C50 + 16LL) == 3 )
    *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x10u;
  if ( a2 )
  {
    v6 = *(_WORD *)(a2 + 8);
    a1 = (__int64 *)v14;
    sub_6F8AB0(a2, (unsigned int)v14, (unsigned int)v15, 0, (unsigned int)&v13, (unsigned int)&v12, 0);
  }
  else
  {
    v6 = word_4F06418[0];
    v13 = *(_QWORD *)&dword_4F063F8;
    v12 = dword_4F06650[0];
    sub_7B8B50(0, 0, a3, a4);
    sub_69ED20((__int64)v15, 0, 10, 0);
  }
  v7 = v12;
  if ( !HIDWORD(qword_4F077B4) )
    return sub_6907F0(a1, v15, v6, &v13, v12, (__int64)a3);
  if ( (_DWORD)qword_4F077B4 )
    return sub_6907F0(a1, v15, v6, &v13, v12, (__int64)a3);
  v9 = *(_BYTE *)(qword_4D03C50 + 19LL);
  if ( (v9 & 0x40) == 0 )
    return sub_6907F0(a1, v15, v6, &v13, v12, (__int64)a3);
  *(_BYTE *)(qword_4D03C50 + 19LL) = v9 & 0xBF;
  sub_6907F0(a1, v15, v6, &v13, v7, (__int64)a3);
  if ( (a3[16] & 0xFD) != 0 )
  {
    v10 = *(_QWORD *)a3;
    for ( i = *(_BYTE *)(*(_QWORD *)a3 + 140LL); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    if ( i )
      sub_6F4D20(a3, 1, 1);
  }
  result = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x40u;
  return result;
}
