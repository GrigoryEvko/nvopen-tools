// Function: sub_D68100
// Address: 0xd68100
//
__int64 __fastcall sub_D68100(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _BOOL4 v4; // r15d
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax

  v4 = 1;
  v6 = a1 + 8;
  if ( !a2 && a3 != v6 )
    v4 = a4[2] < *(_QWORD *)(a3 + 48);
  v7 = sub_22077B0(56);
  *(_QWORD *)(v7 + 32) = 0;
  v8 = v7;
  *(_QWORD *)(v7 + 40) = 0;
  v9 = a4[2];
  *(_QWORD *)(v8 + 48) = v9;
  if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
    sub_BD6050((unsigned __int64 *)(v8 + 32), *a4 & 0xFFFFFFFFFFFFFFF8LL);
  sub_220F040(v4, v8, a3, v6);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
