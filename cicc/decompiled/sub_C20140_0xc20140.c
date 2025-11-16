// Function: sub_C20140
// Address: 0xc20140
//
_BYTE *__fastcall sub_C20140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdi
  char *(*v7)(); // rcx
  char *v8; // rax
  _QWORD v10[4]; // [rsp+0h] [rbp-50h] BYREF
  int v11; // [rsp+20h] [rbp-30h]
  __int64 v12; // [rsp+28h] [rbp-28h]

  v4 = 14;
  v5 = *(_QWORD *)(a1 + 64);
  v6 = *(_QWORD *)(a1 + 72);
  v7 = *(char *(**)())(*(_QWORD *)v6 + 16LL);
  v8 = "Unknown buffer";
  if ( v7 != sub_C1E8B0 )
    v8 = (char *)((__int64 (__fastcall *)(__int64, __int64, __int64))v7)(v6, a2, 14);
  v11 = a2;
  v12 = a3;
  v10[1] = 12;
  v10[0] = &unk_49D9C78;
  v10[2] = v8;
  v10[3] = v4;
  return sub_B6EB20(v5, (__int64)v10);
}
