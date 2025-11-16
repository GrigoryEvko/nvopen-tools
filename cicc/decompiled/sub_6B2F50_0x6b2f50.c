// Function: sub_6B2F50
// Address: 0x6b2f50
//
__int64 __fastcall sub_6B2F50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // bx
  __int64 *v6; // r12
  int v8; // [rsp+4h] [rbp-2FCh] BYREF
  __int64 v9; // [rsp+8h] [rbp-2F8h] BYREF
  _BYTE v10[352]; // [rsp+10h] [rbp-2F0h] BYREF
  __int64 v11[50]; // [rsp+170h] [rbp-190h] BYREF

  if ( a2 )
  {
    v5 = *(_WORD *)(a2 + 8);
    v6 = (__int64 *)v10;
    sub_6F8AB0(a2, (unsigned int)v10, (unsigned int)v11, 0, (unsigned int)&v9, (unsigned int)&v8, 0);
  }
  else
  {
    v6 = a1;
    v5 = word_4F06418[0];
    v9 = *(_QWORD *)&dword_4F063F8;
    v8 = dword_4F06650[0];
    sub_7B8B50(a1, 0, a3, a4);
    sub_69ED20((__int64)v11, 0, 11, 0);
  }
  return sub_69B310(v6, v11, v5, &v9, v8, a3);
}
