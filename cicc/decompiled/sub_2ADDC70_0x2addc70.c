// Function: sub_2ADDC70
// Address: 0x2addc70
//
__int64 __fastcall sub_2ADDC70(__int64 *a1)
{
  __int64 v2; // r12
  unsigned __int8 *v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v8; // r12
  char *v10; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-20h]
  char v12; // [rsp+21h] [rbp-1Fh]

  sub_2AB3250(a1, (void *)byte_3F871B3, 0);
  v2 = a1[60];
  *(_QWORD *)(v2 + 32) = sub_2AB9F30((__int64)a1, a1[31], 1);
  v3 = *(unsigned __int8 **)(a1[60] + 32);
  v12 = 1;
  v10 = "iter.check";
  v11 = 3;
  sub_BD6B50(v3, (const char **)&v10);
  v4 = a1[60];
  *(_QWORD *)(v4 + 40) = sub_2ADD5A0((__int64)a1, a1[31], v5);
  v6 = a1[60];
  *(_QWORD *)(v6 + 48) = sub_2ADCE40((__int64)a1, a1[31]);
  v7 = a1[60];
  *(_QWORD *)(v7 + 24) = sub_2AB9F30((__int64)a1, a1[31], 0);
  v8 = a1[60];
  *(_QWORD *)(v8 + 64) = sub_2AB8740((__int64)a1, a1[30]);
  return a1[30];
}
