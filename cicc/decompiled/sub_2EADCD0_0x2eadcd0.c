// Function: sub_2EADCD0
// Address: 0x2eadcd0
//
unsigned __int64 __fastcall sub_2EADCD0(char *a1, int *a2)
{
  char v2; // al
  int v3; // eax
  _OWORD v5[4]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v6; // [rsp+40h] [rbp-40h]
  __int64 v7; // [rsp+48h] [rbp-38h]
  __int64 v8; // [rsp+50h] [rbp-30h]
  __int64 v9; // [rsp+58h] [rbp-28h]
  __int64 v10; // [rsp+60h] [rbp-20h]
  __int64 v11; // [rsp+68h] [rbp-18h]
  __int64 v12; // [rsp+70h] [rbp-10h]
  void (__fastcall *v13)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v2 = *a1;
  memset(v5, 0, sizeof(v5));
  LOBYTE(v5[0]) = v2;
  v3 = *a2;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = sub_C64CA0;
  *(_DWORD *)((char *)v5 + 1) = v3;
  return sub_AC25F0(v5, 5u, (__int64)sub_C64CA0);
}
