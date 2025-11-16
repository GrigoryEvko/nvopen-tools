// Function: sub_1601410
// Address: 0x1601410
//
__int64 __fastcall sub_1601410(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  char v10[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v11; // [rsp+10h] [rbp-30h]

  v1 = *(a1 - 3);
  v2 = *a1;
  v11 = 257;
  v3 = sub_1648A60(56, 1);
  v4 = v3;
  if ( v3 )
  {
    v5 = v3 - 24;
    sub_15F1EA0(v3, v2, 58, v3 - 24, 1, 0);
    if ( *(_QWORD *)(v4 - 24) )
    {
      v6 = *(_QWORD *)(v4 - 16);
      v7 = *(_QWORD *)(v4 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v7 = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
    }
    *(_QWORD *)(v4 - 24) = v1;
    if ( v1 )
    {
      v8 = *(_QWORD *)(v1 + 8);
      *(_QWORD *)(v4 - 16) = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = (v4 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
      *(_QWORD *)(v4 - 8) = (v1 + 8) | *(_QWORD *)(v4 - 8) & 3LL;
      *(_QWORD *)(v1 + 8) = v5;
    }
    sub_164B780(v4, v10);
  }
  return v4;
}
