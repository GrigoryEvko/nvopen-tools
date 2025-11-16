// Function: sub_305D3D0
// Address: 0x305d3d0
//
__int64 *__fastcall sub_305D3D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  int v8; // [rsp+Ch] [rbp-44h]
  __int64 v9; // [rsp+10h] [rbp-40h]
  __int64 v10; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD *)(a2 + 24);
  v9 = *(_QWORD *)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 32);
  v5 = *(_QWORD *)(a2 + 40);
  v8 = *(_DWORD *)(a2 + 56);
  v6 = sub_22077B0(0x40u);
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = v2;
    *(_QWORD *)(v6 + 16) = v3;
    *(_QWORD *)(v6 + 24) = v4;
    *(_QWORD *)v6 = &unk_4A30850;
    *(_QWORD *)(v6 + 40) = v5;
    *(_QWORD *)(v6 + 32) = v10;
    *(_QWORD *)(v6 + 48) = v9;
    *(_DWORD *)(v6 + 56) = v8;
  }
  *a1 = v6;
  return a1;
}
