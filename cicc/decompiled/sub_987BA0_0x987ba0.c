// Function: sub_987BA0
// Address: 0x987ba0
//
__int64 __fastcall sub_987BA0(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v2 = *((_DWORD *)a2 + 2);
  v13 = v2;
  if ( v2 <= 0x40 )
  {
    v12 = *a2;
LABEL_3:
    v3 = *a2;
    goto LABEL_4;
  }
  sub_C43780(&v12, a2);
  v2 = *((_DWORD *)a2 + 2);
  v11 = v2;
  if ( v2 <= 0x40 )
    goto LABEL_3;
  sub_C43780(&v10, a2);
  v2 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43D10(&v10, a2, v7, v8, v9);
    v2 = v11;
    v4 = v10;
    goto LABEL_6;
  }
  v3 = v10;
LABEL_4:
  v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
  if ( !v2 )
    v4 = 0;
LABEL_6:
  *(_DWORD *)(a1 + 8) = v2;
  v5 = v13;
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 16) = v12;
  return a1;
}
