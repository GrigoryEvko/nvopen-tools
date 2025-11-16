// Function: sub_17CE200
// Address: 0x17ce200
//
__int64 __fastcall sub_17CE200(__int64 *a1, __int64 a2, __int64 **a3, char a4, __int64 *a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rcx
  char v14[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v15; // [rsp+10h] [rbp-30h]

  if ( a3 == *(__int64 ***)a2 )
    return a2;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A4750((__int64 ***)a2, a3, a4);
  v15 = 257;
  v8 = sub_15FE0A0((_QWORD *)a2, (__int64)a3, a4, (__int64)v14, 0);
  v9 = a1[1];
  v10 = v8;
  if ( v9 )
  {
    v11 = (__int64 *)a1[2];
    sub_157E9D0(v9 + 40, v8);
    v12 = *(_QWORD *)(v10 + 24);
    v13 = *v11;
    *(_QWORD *)(v10 + 32) = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v10 + 24) = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v10 + 24;
    *v11 = *v11 & 7 | (v10 + 24);
  }
  sub_164B780(v10, a5);
  sub_12A86E0(a1, v10);
  return v10;
}
