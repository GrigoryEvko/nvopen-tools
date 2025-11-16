// Function: sub_29A87B0
// Address: 0x29a87b0
//
__int64 __fastcall sub_29A87B0(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13[8]; // [rsp+0h] [rbp-40h] BYREF

  v2 = sub_D58140(a1, a2);
  result = 0;
  if ( !v2 )
  {
    v4 = *(__int64 **)(a1 + 8);
    v5 = *v4;
    v6 = v4[1];
    if ( v5 == v6 )
LABEL_16:
      BUG();
    while ( *(_UNKNOWN **)v5 != &unk_4F881C8 )
    {
      v5 += 16;
      if ( v6 == v5 )
        goto LABEL_16;
    }
    v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F881C8);
    v8 = *(__int64 **)(a1 + 8);
    v9 = *(_QWORD *)(v7 + 176);
    v10 = *v8;
    v11 = v8[1];
    if ( v10 == v11 )
LABEL_17:
      BUG();
    while ( *(_UNKNOWN **)v10 != &unk_4F8144C )
    {
      v10 += 16;
      if ( v11 == v10 )
        goto LABEL_17;
    }
    v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(
            *(_QWORD *)(v10 + 8),
            &unk_4F8144C);
    v13[0] = a2;
    v13[1] = v9;
    v13[2] = v12 + 176;
    result = sub_D4B3D0(a2);
    if ( (_BYTE)result )
      return sub_29A8190(v13);
  }
  return result;
}
