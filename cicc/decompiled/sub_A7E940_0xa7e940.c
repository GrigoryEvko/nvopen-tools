// Function: sub_A7E940
// Address: 0xa7e940
//
__int64 __fastcall sub_A7E940(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  int v6; // eax
  _QWORD v8[3]; // [rsp+0h] [rbp-20h] BYREF

  v2 = sub_BCAE30(*(_QWORD *)(a1 + 24));
  v8[1] = v3;
  v8[0] = v2;
  v4 = sub_CA1930(v8);
  v5 = sub_BCCE00(*(_QWORD *)a1, v4);
  v6 = *(_DWORD *)(a1 + 32);
  BYTE4(v8[0]) = *(_BYTE *)(a1 + 8) == 18;
  LODWORD(v8[0]) = v6;
  return sub_BCE1B0(v5, v8[0]);
}
