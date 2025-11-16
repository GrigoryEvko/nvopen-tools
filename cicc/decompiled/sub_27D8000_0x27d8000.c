// Function: sub_27D8000
// Address: 0x27d8000
//
__int64 __fastcall sub_27D8000(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD v18[5]; // [rsp+0h] [rbp-60h] BYREF
  int v19; // [rsp+28h] [rbp-38h]

  v2 = sub_BB98D0((_QWORD *)a1, a2);
  result = 0;
  if ( !v2 )
  {
    v4 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
    if ( v4 && (v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_4F8144C)) != 0 )
      v6 = v5 + 176;
    else
      v6 = 0;
    v7 = *(__int64 **)(a1 + 8);
    v8 = *v7;
    v9 = v7[1];
    if ( v8 == v9 )
LABEL_18:
      BUG();
    while ( *(_UNKNOWN **)v8 != &unk_4F89C28 )
    {
      v8 += 16;
      if ( v9 == v8 )
        goto LABEL_18;
    }
    v10 = *(_DWORD *)(a1 + 172);
    v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F89C28);
    v12 = sub_DFED00(v11, a2);
    v13 = *(__int64 **)(a1 + 8);
    v14 = v12;
    v15 = *v13;
    v16 = v13[1];
    if ( v15 == v16 )
LABEL_19:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F8662C )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_19;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F8662C);
    v18[2] = v6;
    v18[3] = v14;
    v19 = v10;
    v18[0] = sub_CFFAC0(v17, a2);
    v18[1] = 0;
    v18[4] = 0;
    return sub_27D4E70((__int64)v18, a2);
  }
  return result;
}
