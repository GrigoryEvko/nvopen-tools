// Function: sub_15BB120
// Address: 0x15bb120
//
__int64 __fastcall sub_15BB120(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  _QWORD *v6; // rdx
  unsigned int v7; // esi
  int v8; // eax
  int v9; // eax
  __int64 v10[2]; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v11[5]; // [rsp+18h] [rbp-28h] BYREF

  v10[0] = a1;
  if ( (_DWORD)a2 )
  {
    if ( (_DWORD)a2 == 1 )
      sub_1621390(v10[0], a2);
    return v10[0];
  }
  v5 = sub_15B7360(a3, v10, v11);
  v6 = (_QWORD *)v11[0];
  if ( v5 )
    return v10[0];
  v7 = *(_DWORD *)(a3 + 24);
  v8 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v9 = v8 + 1;
  if ( 4 * v9 >= 3 * v7 )
  {
    v7 *= 2;
    goto LABEL_12;
  }
  if ( v7 - *(_DWORD *)(a3 + 20) - v9 <= v7 >> 3 )
  {
LABEL_12:
    sub_15BAB20(a3, v7);
    sub_15B7360(a3, v10, v11);
    v6 = (_QWORD *)v11[0];
    v9 = *(_DWORD *)(a3 + 16) + 1;
  }
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v6 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v6 = v10[0];
  return v10[0];
}
