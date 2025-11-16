// Function: sub_C53200
// Address: 0xc53200
//
__int64 __fastcall sub_C53200(__int64 a1, __int64 a2, __int64 a3)
{
  volatile signed __int32 *v4; // rdi
  volatile signed __int32 *v6; // [rsp+8h] [rbp-18h] BYREF

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  sub_CA41E0(&v6);
  v4 = v6;
  *(_QWORD *)(a1 + 16) = v6;
  if ( v4 && !_InterlockedSub(v4 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v4 + 8LL))(v4);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_WORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 58) = 0;
  return 0;
}
