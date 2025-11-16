// Function: sub_398B900
// Address: 0x398b900
//
unsigned __int64 __fastcall sub_398B900(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r15d
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  __int64 v7; // [rsp+8h] [rbp-48h]
  unsigned __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v3 = *(_DWORD *)(a2 + 600);
  v7 = *(_QWORD *)(a1 + 8);
  v4 = sub_22077B0(0x3A8u);
  v5 = v4;
  if ( v4 )
    sub_39C7990(v4, v3, v2, v7, a1, a1 + 4520);
  *(_QWORD *)(v5 + 56) = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 88);
  sub_39C7CA0(v5);
  if ( *(_BYTE *)(a1 + 4514) )
    sub_39A3E90(v5);
  v8[0] = v5;
  sub_398B840((_QWORD *)a1, a2, v5 + 8, (__int64 *)v8);
  if ( v8[0] )
    sub_3985790(v8[0]);
  return v5;
}
