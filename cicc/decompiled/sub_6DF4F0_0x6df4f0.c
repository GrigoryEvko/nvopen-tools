// Function: sub_6DF4F0
// Address: 0x6df4f0
//
__int64 __fastcall sub_6DF4F0(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // [rsp+8h] [rbp-B8h]
  int v10; // [rsp+10h] [rbp-B0h]
  int v11; // [rsp+2Ch] [rbp-94h] BYREF
  _BYTE v12[144]; // [rsp+30h] [rbp-90h] BYREF

  v2 = *a1;
  v3 = *a1 + 24LL * a2;
  v4 = *(_QWORD *)(v3 + 8);
  if ( !v4 )
  {
    v6 = *(_QWORD *)(v3 + 16);
    v4 = *(_QWORD *)(v6 + 64);
    if ( (*(_DWORD *)v3 & 0xFFFFFFFC) != 0xFFFFFFFC )
    {
      v7 = **(_QWORD **)(v6 + 56);
      v11 = 0;
      v8 = **(_QWORD **)(*(_QWORD *)(v7 + 88) + 32LL);
      v9 = **(_QWORD **)(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)(v2 + 24LL * (*(_DWORD *)v3 >> 2) + 16) + 56LL) + 88LL)
                       + 32LL);
      v10 = sub_6DF4F0();
      sub_892150(v12);
      v4 = sub_8A55D0(v7, v4, v8, 0, v10, v9, v6 + 28, 0, (__int64)&v11, (__int64)v12);
    }
    *(_QWORD *)(v3 + 8) = v4;
  }
  return v4;
}
