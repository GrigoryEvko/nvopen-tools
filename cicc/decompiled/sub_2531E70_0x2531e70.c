// Function: sub_2531E70
// Address: 0x2531e70
//
__int64 __fastcall sub_2531E70(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  int v5; // eax
  int v6; // r14d
  unsigned int v7; // r12d
  unsigned int v8; // r12d
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+10h] [rbp-30h] BYREF

  v13[1] = a1;
  v1 = sub_C996C0("Attributor::run", 15, 0, 0);
  v13[0] = &unk_4A16C28;
  if ( byte_4FEE728 )
  {
    v10 = *(_QWORD *)(a1 + 200);
    v14[1] = a1;
    v11 = *(_QWORD *)(v10 + 32);
    v12 = v11 + 8LL * *(unsigned int *)(v10 + 40);
    for ( v14[0] = v11; v14[0] != v12; v14[0] += 8LL )
      sub_25A1010(v14);
  }
  *(_DWORD *)(a1 + 3552) = 1;
  sub_251CD10(a1);
  if ( byte_4FEEBA8 )
    sub_2531A30(a1 + 216);
  if ( byte_4FEE9C8 )
    sub_2531960(a1 + 216);
  if ( (_BYTE)qword_4FEE8E8 )
    sub_250FFC0(a1 + 216, 15, v2, v3, v4);
  *(_DWORD *)(a1 + 3552) = 2;
  v5 = sub_25226A0(a1);
  *(_DWORD *)(a1 + 3552) = 3;
  v6 = v5;
  v7 = sub_2524E70(a1);
  if ( byte_4FEE728 )
    sub_25A1A10(v13);
  v8 = sub_250C0B0(v6, v7);
  if ( v1 )
    sub_C9AF60(v1);
  return v8;
}
