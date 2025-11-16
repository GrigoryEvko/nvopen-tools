// Function: sub_2EC2940
// Address: 0x2ec2940
//
__int64 __fastcall sub_2EC2940(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 *v8; // rdi
  __int64 result; // rax
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int128 v13; // [rsp-20h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-28h]
  __int64 v15; // [rsp+10h] [rbp-20h]

  v7 = *(__int64 **)(a1 + 48);
  v8 = *(__int64 **)(a1 + 40);
  if ( v8 == v7 )
    return 0;
  if ( (char *)v7 - (char *)v8 > 8 )
  {
    v10 = *(_QWORD *)(a1 + 16);
    v11 = v7 - 1;
    v12 = *v11;
    LOBYTE(v15) = *(_BYTE *)(a1 + 32);
    v14 = *(_QWORD *)(a1 + 24);
    *v11 = *v8;
    *((_QWORD *)&v13 + 1) = v14;
    *(_QWORD *)&v13 = v10;
    sub_2EC2730((__int64)v8, 0, v11 - v8, v12, v14, a6, v13, v15);
    v7 = *(__int64 **)(a1 + 48);
  }
  result = *(v7 - 1);
  *(_QWORD *)(a1 + 48) = v7 - 1;
  *a2 = 0;
  return result;
}
