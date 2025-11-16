// Function: sub_C41F80
// Address: 0xc41f80
//
unsigned __int64 __fastcall sub_C41F80(__int64 a1)
{
  char v1; // al
  char v2; // dl
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 *v5; // rax
  int *v6; // rdx
  char v7; // cl
  int *v9; // rdx
  bool v10; // cl
  char v11; // [rsp+6h] [rbp-1Ah] BYREF
  char v12; // [rsp+7h] [rbp-19h] BYREF
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_BYTE *)(a1 + 20);
  v2 = v1 & 7;
  if ( (v1 & 7) == 1 )
  {
    v10 = 0;
    v9 = (int *)(*(_QWORD *)a1 + 8LL);
  }
  else
  {
    if ( v2 && v2 != 3 )
    {
      v3 = sub_C33930(a1);
      v4 = v3 + 8LL * (unsigned int)sub_C337D0(a1);
      v5 = (__int64 *)sub_C33930(a1);
      v13[0] = sub_AF66D0(v5, v4);
      v6 = (int *)(*(_QWORD *)a1 + 8LL);
      v7 = *(_BYTE *)(a1 + 20) >> 3;
      v12 = *(_BYTE *)(a1 + 20) & 7;
      v11 = v7 & 1;
      return sub_C41DF0(&v12, &v11, v6, (int *)(a1 + 16), v13);
    }
    v9 = (int *)(*(_QWORD *)a1 + 8LL);
    v10 = (v1 & 8) != 0;
  }
  LOBYTE(v13[0]) = v10;
  v12 = v1 & 7;
  return sub_C41D60(&v12, (char *)v13, v9);
}
