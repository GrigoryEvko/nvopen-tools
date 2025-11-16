// Function: sub_987D70
// Address: 0x987d70
//
__int64 __fastcall sub_987D70(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v4; // ebx
  __int64 v5; // r15
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rdx
  __int64 v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-38h]

  v4 = *((_DWORD *)a2 + 6);
  v15 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = a2[2];
LABEL_3:
    v6 = a3[2] | v5;
    v14 = v6;
    goto LABEL_4;
  }
  sub_C43780(&v14, a2 + 2);
  v4 = v15;
  if ( v15 <= 0x40 )
  {
    v5 = v14;
    goto LABEL_3;
  }
  sub_C43BD0(&v14, a3 + 2);
  v4 = v15;
  v6 = v14;
LABEL_4:
  v7 = *((_DWORD *)a2 + 2);
  v15 = 0;
  v13 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780(&v12, a2);
    v7 = v13;
    if ( v13 > 0x40 )
    {
      sub_C43BD0(&v12, a3);
      v7 = v13;
      v10 = v12;
      v9 = v15;
      goto LABEL_7;
    }
    v8 = v12;
    v9 = v15;
  }
  else
  {
    v8 = *a2;
    v9 = 0;
  }
  v10 = *a3 | v8;
LABEL_7:
  *(_DWORD *)(a1 + 8) = v7;
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 24) = v4;
  *(_QWORD *)(a1 + 16) = v6;
  if ( v9 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return a1;
}
