// Function: sub_12456E0
// Address: 0x12456e0
//
__int64 __fastcall sub_12456E0(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // rdx
  unsigned __int64 v3; // r14
  unsigned int v4; // r12d
  char v6; // [rsp+Ah] [rbp-56h] BYREF
  char v7; // [rsp+Bh] [rbp-55h] BYREF
  int v8; // [rsp+Ch] [rbp-54h] BYREF
  int v9; // [rsp+10h] [rbp-50h] BYREF
  int v10; // [rsp+14h] [rbp-4Ch] BYREF
  int v11; // [rsp+18h] [rbp-48h] BYREF
  int v12; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v13[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+30h] [rbp-30h] BYREF

  v1 = *(_BYTE **)(a1 + 248);
  v2 = *(_QWORD *)(a1 + 256);
  v13[0] = (__int64)v14;
  v3 = *(_QWORD *)(a1 + 232);
  sub_12060D0(v13, v1, (__int64)&v1[v2]);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' in global variable")
    || (unsigned __int8)sub_120C500(a1, (__int64)&v8, &v6, &v9, &v10, (unsigned __int8 *)&v7)
    || (unsigned __int8)sub_120C1C0(a1, &v11)
    || (unsigned __int8)sub_120A6C0(a1, &v12) )
  {
    v4 = 1;
  }
  else if ( (unsigned int)(*(_DWORD *)(a1 + 240) - 98) <= 1 )
  {
    v4 = sub_1243EC0(a1, v13, 0xFFFFFFFF, v3, v8, v9, v10, v7, v11, v12);
  }
  else
  {
    v4 = sub_1244A70(a1, (char *)v13, 0xFFFFFFFF, v3, v8, v6, v9, v10, v7, v11, v12);
  }
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
  return v4;
}
