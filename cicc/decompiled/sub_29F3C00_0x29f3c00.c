// Function: sub_29F3C00
// Address: 0x29f3c00
//
_QWORD *__fastcall sub_29F3C00(__int64 *a1, char *a2, signed __int64 a3, char a4, _DWORD a5, _DWORD a6, char a7)
{
  __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v10; // rbx
  _QWORD *v11; // r12
  __int64 v13; // [rsp+8h] [rbp-38h]

  v8 = sub_AC9B20(*a1, a2, a3, 1);
  BYTE4(v13) = 0;
  v9 = *(_QWORD **)(v8 + 8);
  v10 = v8;
  v11 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v11 )
    sub_B30000((__int64)v11, (__int64)a1, v9, 1, 8, v10, (__int64)&a7, 0, 0, v13, 0);
  if ( a4 )
    *((_BYTE *)v11 + 32) = v11[4] & 0x3F | 0x80;
  sub_B2F770((__int64)v11, 0);
  return v11;
}
