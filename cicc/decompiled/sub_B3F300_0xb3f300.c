// Function: sub_B3F300
// Address: 0xb3f300
//
unsigned __int64 __fastcall sub_B3F300(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  char v6; // si
  int v7; // edx
  __int64 v8; // r8
  char v9; // cl
  __int64 v10; // rdi
  int v12; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v13; // [rsp+10h] [rbp-70h] BYREF
  __int64 v14[2]; // [rsp+18h] [rbp-68h] BYREF
  __int64 v15[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v16; // [rsp+38h] [rbp-48h] BYREF
  char v17; // [rsp+40h] [rbp-40h] BYREF
  char v18[3]; // [rsp+41h] [rbp-3Fh] BYREF
  int v19; // [rsp+44h] [rbp-3Ch] BYREF
  char v20[56]; // [rsp+48h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 24);
  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 56);
  v4 = *(_QWORD *)(a1 + 64);
  v5 = sub_B3B7D0(a1);
  v6 = *(_BYTE *)(a1 + 96);
  v7 = *(_DWORD *)(a1 + 100);
  v8 = v5;
  LOBYTE(v5) = *(_BYTE *)(a1 + 104);
  v9 = *(_BYTE *)(a1 + 97);
  v15[1] = v4;
  v10 = *(_QWORD *)(a1 + 8);
  v16 = v8;
  v20[0] = v5;
  v13 = v10;
  v17 = v6;
  v18[0] = v9;
  v19 = v7;
  v14[0] = v1;
  v14[1] = v2;
  v15[0] = v3;
  v12 = sub_B3B830(v14, v15, &v17, v18, &v19, &v16, v20);
  return sub_B3BD80(&v13, &v12);
}
