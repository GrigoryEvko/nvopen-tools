// Function: sub_E1FE40
// Address: 0xe1fe40
//
__int64 __fastcall sub_E1FE40(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // rax
  char v14; // dl

  v1 = sub_E12DE0(a1);
  if ( v1 )
  {
    v6 = v1;
    if ( *a1 == a1[1] || *(_BYTE *)*a1 != 73 )
      return v6;
    v12 = sub_E1F700((__int64)a1, 0, v2, v3, v4, v5);
    if ( v12 )
    {
      v13 = sub_E0E790((__int64)(a1 + 102), 32, v8, v9, v10, v11);
      if ( v13 )
      {
        *(_QWORD *)(v13 + 16) = v6;
        v6 = v13;
        *(_WORD *)(v13 + 8) = 16429;
        v14 = *(_BYTE *)(v13 + 10);
        *(_QWORD *)(v13 + 24) = v12;
        *(_BYTE *)(v13 + 10) = v14 & 0xF0 | 5;
        *(_QWORD *)v13 = &unk_49DFEA8;
        return v6;
      }
    }
  }
  return 0;
}
