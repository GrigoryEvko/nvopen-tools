// Function: sub_3446D80
// Address: 0x3446d80
//
__int64 __fastcall sub_3446D80(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // ecx
  __int64 v5; // rax
  int v6; // ecx
  unsigned int v7; // edx
  __int64 v8; // rax
  int v9; // eax
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  int v14; // [rsp+18h] [rbp-28h]

  v4 = *(_DWORD *)(a3 + 24);
  if ( (unsigned int)(v4 + *(_DWORD *)(a2 + 24)) > 0x40 )
  {
    sub_C44A70((__int64)&v13, a2 + 16, a3 + 16);
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 16);
    v14 = v4 + *(_DWORD *)(a2 + 24);
    v13 = *(_QWORD *)(a3 + 16) | (v5 << v4);
  }
  v6 = *(_DWORD *)(a3 + 8);
  v7 = v6 + *(_DWORD *)(a2 + 8);
  if ( v7 > 0x40 )
  {
    sub_C44A70((__int64)&v11, a2, a3);
    v7 = v12;
    v8 = v11;
  }
  else
  {
    v8 = *(_QWORD *)a3 | (*(_QWORD *)a2 << v6);
  }
  *(_QWORD *)a1 = v8;
  v9 = v14;
  *(_DWORD *)(a1 + 8) = v7;
  *(_DWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a1 + 16) = v13;
  return a1;
}
