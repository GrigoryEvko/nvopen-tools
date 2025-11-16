// Function: sub_1E6ECD0
// Address: 0x1e6ecd0
//
__int64 __fastcall sub_1E6ECD0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_22077B0(64);
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)v1 = off_49FCB18;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_BYTE *)(v1 + 32) = 1;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 56) = 0;
  }
  v5[0] = v1;
  v2 = sub_22077B0(4072);
  v3 = v2;
  if ( v2 )
    sub_1E6E680(v2, a1, v5);
  if ( v5[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5[0] + 16LL))(v5[0]);
  return v3;
}
