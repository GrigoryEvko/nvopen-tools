// Function: sub_1A2F250
// Address: 0x1a2f250
//
__int64 __fastcall sub_1A2F250(__int64 a1, __int64 a2)
{
  __int64 ***v2; // r12
  __int64 result; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 ***v7; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(__int64 ****)a2;
  result = sub_1599EF0(**(__int64 ****)a2);
  if ( *(_QWORD *)a2 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    v5 = *(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v5 = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(v4 + 16) & 3LL | v5;
  }
  *(_QWORD *)a2 = result;
  if ( result )
  {
    v6 = *(_QWORD *)(result + 8);
    *(_QWORD *)(a2 + 8) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = (a2 + 8) | *(_QWORD *)(v6 + 16) & 3LL;
    *(_QWORD *)(a2 + 16) = (result + 8) | *(_QWORD *)(a2 + 16) & 3LL;
    *(_QWORD *)(result + 8) = a2;
  }
  if ( *((_BYTE *)v2 + 16) > 0x17u )
  {
    v7 = v2;
    result = sub_1AE9990(v2, 0);
    if ( (_BYTE)result )
      return sub_1A2EDE0(a1 + 208, (__int64 *)&v7);
  }
  return result;
}
