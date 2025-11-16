// Function: sub_22C1AA0
// Address: 0x22c1aa0
//
__int64 __fastcall sub_22C1AA0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // [rsp+0h] [rbp-50h] BYREF
  __int64 v7; // [rsp+8h] [rbp-48h]
  __int64 v8; // [rsp+10h] [rbp-40h]
  __int64 v9; // [rsp+20h] [rbp-30h]
  __int64 v10; // [rsp+28h] [rbp-28h]
  __int64 v11; // [rsp+30h] [rbp-20h]

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v6 = 0;
    v2 = (_QWORD *)(a1 + 16);
    v3 = 6;
    v7 = 0;
    v8 = -4096;
    v9 = 0;
    v10 = 0;
    v11 = -8192;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)result )
      return result;
    v6 = 0;
    v2 = *(_QWORD **)(a1 + 16);
    v7 = 0;
    v3 = 3 * result;
    v8 = -4096;
    v9 = 0;
    v10 = 0;
    v11 = -8192;
  }
  v4 = &v2[v3];
  do
  {
    v5 = v2[2];
    if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
      sub_BD60C0(v2);
    v2 += 3;
  }
  while ( v2 != v4 );
  result = v8;
  if ( v8 != -4096 && v8 != 0 )
    return sub_BD60C0(&v6);
  return result;
}
