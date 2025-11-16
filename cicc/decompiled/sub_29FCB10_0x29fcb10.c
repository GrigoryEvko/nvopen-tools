// Function: sub_29FCB10
// Address: 0x29fcb10
//
_QWORD *__fastcall sub_29FCB10(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // r13
  unsigned __int8 *v6; // rax
  __int16 v7; // dx
  unsigned __int64 *v8; // r8
  unsigned __int8 v9; // al
  char v10; // dl
  __int64 v11; // rcx
  const char *v13[4]; // [rsp+0h] [rbp-50h] BYREF
  char v14; // [rsp+20h] [rbp-30h]
  char v15; // [rsp+21h] [rbp-2Fh]

  v13[0] = (const char *)sub_BD5C60((__int64)a2);
  v4 = sub_B8C340(v13);
  v5 = *(unsigned __int8 **)(sub_F38250(a3, a2 + 3, 0, 0, v4, *(_QWORD *)(a1 + 8), 0, 0) + 40);
  v15 = 1;
  v13[0] = "cdce.call";
  v14 = 3;
  sub_BD6B50(v5, v13);
  v6 = (unsigned __int8 *)sub_AA56F0((__int64)v5);
  v15 = 1;
  v14 = 3;
  v13[0] = "cdce.end";
  sub_BD6B50(v6, v13);
  sub_B43D10(a2);
  v8 = (unsigned __int64 *)sub_AA5190((__int64)v5);
  if ( v8 )
  {
    v9 = v7;
    v10 = HIBYTE(v7);
  }
  else
  {
    v10 = 0;
    v9 = 0;
  }
  v11 = v9;
  BYTE1(v11) = v10;
  return sub_B44240(a2, (__int64)v5, v8, v11);
}
