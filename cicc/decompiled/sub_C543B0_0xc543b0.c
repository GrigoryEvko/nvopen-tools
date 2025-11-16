// Function: sub_C543B0
// Address: 0xc543b0
//
__int64 __fastcall sub_C543B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // ecx
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = sub_CB7210(a1, a2);
  v4 = *(_QWORD *)(a1 + 24);
  v10[2] = 2;
  v5 = v3;
  v6 = *(_QWORD *)(a1 + 32);
  v10[0] = v4;
  v10[1] = v6;
  sub_C51AE0(v5, (__int64)v10);
  v7 = *(_QWORD *)(a1 + 32);
  if ( v7 == 1 )
    v8 = qword_4C5C728 + 6;
  else
    v8 = v7 + qword_4C5C718 + 5;
  return sub_C540D0(*(_OWORD *)(a1 + 40), a2, v8);
}
