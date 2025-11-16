// Function: sub_9692E0
// Address: 0x9692e0
//
__int64 __fastcall sub_9692E0(__int64 a1, __int64 *a2)
{
  unsigned int v3; // eax
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int64 v12; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-18h]

  v3 = *((_DWORD *)a2 + 2);
  v4 = *a2;
  v5 = 1LL << ((unsigned __int8)v3 - 1);
  if ( v3 <= 0x40 )
  {
    v6 = *a2;
    if ( (v5 & v4) == 0 )
    {
      *(_DWORD *)(a1 + 8) = v3;
      *(_QWORD *)a1 = v4;
      return a1;
    }
    v13 = *((_DWORD *)a2 + 2);
    goto LABEL_7;
  }
  if ( (*(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6)) & v5) == 0 )
  {
    *(_DWORD *)(a1 + 8) = v3;
    sub_C43780(a1, a2);
    return a1;
  }
  v13 = *((_DWORD *)a2 + 2);
  sub_C43780(&v12, a2);
  v3 = v13;
  if ( v13 <= 0x40 )
  {
    v6 = v12;
LABEL_7:
    v8 = ~v6 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v3);
    if ( !v3 )
      v8 = 0;
    v12 = v8;
    goto LABEL_10;
  }
  sub_C43D10(&v12, a2, v9, v10, v11);
LABEL_10:
  sub_C46250(&v12);
  *(_DWORD *)(a1 + 8) = v13;
  *(_QWORD *)a1 = v12;
  return a1;
}
