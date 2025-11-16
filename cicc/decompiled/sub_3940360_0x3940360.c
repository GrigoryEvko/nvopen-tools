// Function: sub_3940360
// Address: 0x3940360
//
__int64 __fastcall sub_3940360(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  int v4; // eax
  __int64 v5; // rdx
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 v13; // [rsp+8h] [rbp-28h]
  char v14; // [rsp+10h] [rbp-20h]

  sub_3940120((__int64)&v12, a2);
  if ( (v14 & 1) != 0 )
  {
    v4 = v12;
    v5 = v13;
    if ( v12 )
      goto LABEL_3;
  }
  v7 = a2[11];
  v8 = (a2[12] - v7) >> 5;
  if ( v12 >= v8 )
  {
    v5 = sub_393D180((__int64)&v12, (__int64)a2, v7, v8, v2, v3);
    v4 = 8;
LABEL_3:
    *(_DWORD *)a1 = v4;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v5;
    return a1;
  }
  v9 = (__int64 *)(v7 + 32LL * v12);
  v10 = *v9;
  v11 = v9[1];
  *(_BYTE *)(a1 + 16) &= ~1u;
  *(_QWORD *)(a1 + 8) = v11;
  *(_QWORD *)a1 = v10;
  return a1;
}
