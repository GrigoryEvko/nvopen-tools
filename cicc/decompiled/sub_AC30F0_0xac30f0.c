// Function: sub_AC30F0
// Address: 0xac30f0
//
bool __fastcall sub_AC30F0(__int64 a1)
{
  char v1; // dl
  unsigned int v2; // ebx
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  char v8; // [rsp+Fh] [rbp-71h]
  _QWORD v9[4]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v10[80]; // [rsp+30h] [rbp-50h] BYREF

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 17 )
  {
    v2 = *(_DWORD *)(a1 + 32);
    if ( v2 <= 0x40 )
      return *(_QWORD *)(a1 + 24) == 0;
    else
      return v2 == (unsigned int)sub_C444A0(a1 + 24);
  }
  else if ( v1 == 18 )
  {
    v4 = sub_C33320(a1);
    sub_C3B1B0(v10, 0.0);
    sub_C407B0(v9, v10, v4);
    sub_C338F0(v10);
    sub_C41640(v9, *(_QWORD *)(a1 + 24), 1, v10);
    v8 = sub_AC3090(a1, v9, v5, v6, v7);
    sub_91D830(v9);
    return v8;
  }
  else
  {
    return v1 == 14 || (unsigned __int8)(v1 - 19) <= 2u;
  }
}
