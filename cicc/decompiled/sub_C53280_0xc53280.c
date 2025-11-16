// Function: sub_C53280
// Address: 0xc53280
//
__int64 __fastcall sub_C53280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD v13[8]; // [rsp+0h] [rbp-40h] BYREF

  v5 = a3;
  v6 = a5;
  if ( !a3 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    a4 = *(_QWORD *)(a1 + 32);
  }
  if ( a4 )
  {
    if ( !qword_4F83CE0 )
      sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
    v11 = sub_CB6200(v6, *(_QWORD *)qword_4F83CE0, *(_QWORD *)(qword_4F83CE0 + 8));
    v12 = sub_904010(v11, ": for the ");
    v13[0] = v5;
    v13[1] = a4;
    v13[2] = 0;
    sub_C51AE0(v12, (__int64)v13);
  }
  else
  {
    sub_A51340(a5, *(const void **)(a1 + 40), *(_QWORD *)(a1 + 48));
  }
  v8 = *(_QWORD *)(v6 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v8) <= 8 )
  {
    v6 = sub_CB6200(v6, " option: ", 9);
  }
  else
  {
    *(_BYTE *)(v8 + 8) = 32;
    *(_QWORD *)v8 = 0x3A6E6F6974706F20LL;
    *(_QWORD *)(v6 + 32) += 9LL;
  }
  sub_CA0E80(a2, v6);
  v9 = *(_BYTE **)(v6 + 32);
  if ( *(_BYTE **)(v6 + 24) == v9 )
  {
    sub_CB6200(v6, "\n", 1);
  }
  else
  {
    *v9 = 10;
    ++*(_QWORD *)(v6 + 32);
  }
  return 1;
}
