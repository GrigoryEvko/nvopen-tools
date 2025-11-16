// Function: sub_15EE310
// Address: 0x15ee310
//
unsigned __int64 __fastcall sub_15EE310(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  char v6; // cl
  __int64 v7; // rsi
  char v8; // dl
  __int64 v9; // r8
  int v11; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  __int64 v13[2]; // [rsp+18h] [rbp-68h] BYREF
  __int64 v14[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v15; // [rsp+38h] [rbp-48h] BYREF
  char v16; // [rsp+40h] [rbp-40h] BYREF
  char v17[3]; // [rsp+41h] [rbp-3Fh] BYREF
  int v18[15]; // [rsp+44h] [rbp-3Ch] BYREF

  v1 = *(_QWORD *)(a1 + 24);
  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 56);
  v4 = *(_QWORD *)(a1 + 64);
  v5 = sub_15EAB70(a1);
  v6 = *(_BYTE *)(a1 + 96);
  v7 = *(_QWORD *)a1;
  v8 = *(_BYTE *)(a1 + 97);
  v9 = v5;
  LODWORD(v5) = *(_DWORD *)(a1 + 100);
  v14[1] = v4;
  v12 = v7;
  v15 = v9;
  v16 = v6;
  v17[0] = v8;
  v13[0] = v1;
  v13[1] = v2;
  v14[0] = v3;
  v18[0] = v5;
  v11 = sub_15EAB80(v13, v14, &v16, v17, v18, &v15);
  return sub_15EDB00(&v12, &v11);
}
