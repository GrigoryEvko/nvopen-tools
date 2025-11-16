// Function: sub_2EC5320
// Address: 0x2ec5320
//
__int64 __fastcall sub_2EC5320(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_22077B0(0x40u);
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)v1 = off_4A2A088;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_BYTE *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 56) = 0;
  }
  v5[0] = v1;
  v2 = sub_22077B0(0x1A08u);
  v3 = v2;
  if ( v2 )
    sub_2EC4D40(v2, a1, v5);
  if ( v5[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5[0] + 16LL))(v5[0]);
  return v3;
}
