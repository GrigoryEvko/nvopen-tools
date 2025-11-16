// Function: sub_C41D60
// Address: 0xc41d60
//
unsigned __int64 __fastcall sub_C41D60(char *a1, char *a2, int *a3)
{
  char v3; // al
  char v4; // al
  int v5; // eax
  _OWORD v7[4]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v8; // [rsp+40h] [rbp-40h]
  __int64 v9; // [rsp+48h] [rbp-38h]
  __int64 v10; // [rsp+50h] [rbp-30h]
  __int64 v11; // [rsp+58h] [rbp-28h]
  __int64 v12; // [rsp+60h] [rbp-20h]
  __int64 v13; // [rsp+68h] [rbp-18h]
  __int64 v14; // [rsp+70h] [rbp-10h]
  __int64 (__fastcall *v15)(); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  memset(v7, 0, sizeof(v7));
  LOBYTE(v7[0]) = v3;
  v4 = *a2;
  v8 = 0;
  BYTE1(v7[0]) = v4;
  v5 = *a3;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = sub_C64CA0;
  *(_DWORD *)((char *)v7 + 2) = v5;
  return sub_AC25F0(v7, 6u, (__int64)sub_C64CA0);
}
