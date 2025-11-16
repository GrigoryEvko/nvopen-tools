// Function: sub_2EE6450
// Address: 0x2ee6450
//
__int64 __fastcall sub_2EE6450(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 *v8; // r14
  __int64 (*v9)(void); // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  int v17; // eax
  __int64 v19; // [rsp+8h] [rbp-38h]

  v2 = sub_BB98D0(a1, *(_QWORD *)a2);
  if ( (_BYTE)v2 )
    return 0;
  v3 = v2;
  v19 = 0;
  v8 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v9 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v9 != sub_2DAC790 )
    v19 = v9();
  sub_2ED4FB0((__int64)(a1 + 25), (__int64)v8, v4, v5, v6, v7);
  sub_2ED4FB0((__int64)(a1 + 35), (__int64)v8, v10, v11, v12, v13);
  v16 = *(_QWORD *)(a2 + 328);
  if ( v16 == a2 + 320 )
  {
    return 0;
  }
  else
  {
    do
    {
      v17 = sub_2EE40E0((__int64)a1, v16, v8, v19, v14, v15);
      v16 = *(_QWORD *)(v16 + 8);
      v3 |= v17;
    }
    while ( a2 + 320 != v16 );
  }
  return v3;
}
