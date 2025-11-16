// Function: sub_987AA0
// Address: 0x987aa0
//
__int64 __fastcall sub_987AA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v7; // eax
  __int64 v8; // [rsp+0h] [rbp-40h] BYREF
  int v9; // [rsp+8h] [rbp-38h]
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  int v11; // [rsp+18h] [rbp-28h]

  v3 = a3;
  v4 = *(_DWORD *)(a2 + 8);
  if ( (unsigned int)a3 > v4 )
  {
    sub_C449B0(&v10, a2 + 16, a3);
    sub_C449B0(&v8, a2, v3);
    goto LABEL_8;
  }
  if ( (unsigned int)a3 < v4 )
  {
    sub_C44740(&v10, a2 + 16);
    sub_C44740(&v8, a2);
LABEL_8:
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 24) = v11;
    *(_QWORD *)(a1 + 16) = v10;
    return a1;
  }
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780(a1, a2);
    v7 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 <= 0x40 )
      goto LABEL_5;
LABEL_11:
    sub_C43780(a1 + 16, a2 + 16);
    return a1;
  }
  *(_QWORD *)a1 = *(_QWORD *)a2;
  v5 = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v5;
  if ( v5 > 0x40 )
    goto LABEL_11;
LABEL_5:
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  return a1;
}
