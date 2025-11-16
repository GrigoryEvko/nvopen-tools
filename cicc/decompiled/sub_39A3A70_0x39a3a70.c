// Function: sub_39A3A70
// Address: 0x39a3a70
//
__int64 __fastcall sub_39A3A70(__int64 a1, __int64 a2, __int16 a3, __int64 a4)
{
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r15
  __int64 v10[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = sub_3981ED0(a2);
  v6 = sub_3981ED0(a4);
  v7 = v6;
  if ( !v5 )
  {
    v5 = sub_3981ED0(a1 + 8);
    if ( v7 )
      goto LABEL_3;
LABEL_5:
    v7 = sub_3981ED0(a1 + 8);
    goto LABEL_3;
  }
  if ( !v6 )
    goto LABEL_5;
LABEL_3:
  v10[1] = a4;
  LODWORD(v10[0]) = 6;
  WORD2(v10[0]) = a3;
  HIWORD(v10[0]) = 3 * (v5 == v7) + 16;
  return sub_39A31C0((__int64 *)(a2 + 8), (__int64 *)(a1 + 88), v10);
}
