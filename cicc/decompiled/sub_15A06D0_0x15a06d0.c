// Function: sub_15A06D0
// Address: 0x15a06d0
//
__int64 __fastcall sub_15A06D0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-58h]
  _BYTE v17[8]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v18[9]; // [rsp+18h] [rbp-48h] BYREF

  switch ( *((_BYTE *)a1 + 8) )
  {
    case 0:
    case 7:
    case 8:
    case 9:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0x10:
      return sub_1598F00(a1);
    case 1:
      v5 = sub_1698260(a1, a2, a3, a4);
      goto LABEL_5;
    case 2:
      v5 = sub_1698270(a1, a2);
      goto LABEL_5;
    case 3:
      v5 = sub_1698280(a1);
      goto LABEL_5;
    case 4:
      v5 = sub_16982A0();
      goto LABEL_5;
    case 5:
      v5 = sub_1698290();
LABEL_5:
      v8 = v5;
      v9 = sub_16982C0(a1, a2, v6, v7);
      v10 = v9;
      if ( v8 == v9 )
        sub_169C580(v18, v9, 0);
      else
        sub_1698390(v18, v8, 0);
      if ( v18[0] == v10 )
        sub_169C980(v18, 0);
      else
        sub_169B620(v18, 0);
      v11 = sub_159CCF0(*a1, (__int64)v17);
      sub_127D120(v18);
      goto LABEL_10;
    case 6:
      v16 = 128;
      sub_16A4EF0(&v15, 0, 0);
      v14 = sub_16982C0(&v15, 0, v12, v13);
      sub_169D060(v18, v14, &v15);
      v11 = sub_159CCF0(*a1, (__int64)v17);
      sub_127D120(v18);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
LABEL_10:
      result = v11;
      break;
    case 0xA:
      result = sub_1594470(*a1);
      break;
    case 0xB:
      result = sub_15A0680((__int64)a1, 0, 0);
      break;
    case 0xF:
      result = sub_1599A20(a1);
      break;
  }
  return result;
}
