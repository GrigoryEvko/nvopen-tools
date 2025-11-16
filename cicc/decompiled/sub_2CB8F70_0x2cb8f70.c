// Function: sub_2CB8F70
// Address: 0x2cb8f70
//
__int64 __fastcall sub_2CB8F70(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax

  v2 = sub_BB98D0((_QWORD *)a1, a2);
  if ( (_BYTE)v2 )
  {
    return 0;
  }
  else
  {
    v5 = *(__int64 **)(a1 + 8);
    v3 = v2;
    v6 = *v5;
    v7 = v5[1];
    if ( v6 == v7 )
LABEL_22:
      BUG();
    while ( *(_UNKNOWN **)v6 != &unk_4F8144C )
    {
      v6 += 16;
      if ( v7 == v6 )
        goto LABEL_22;
    }
    v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F8144C);
    v9 = *(__int64 **)(a1 + 8);
    v10 = v8 + 176;
    v11 = *v9;
    v12 = v9[1];
    if ( v11 == v12 )
LABEL_23:
      BUG();
    while ( *(_UNKNOWN **)v11 != &unk_4F8FBD4 )
    {
      v11 += 16;
      if ( v12 == v11 )
        goto LABEL_23;
    }
    v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
            *(_QWORD *)(v11 + 8),
            &unk_4F8FBD4);
    if ( (_BYTE)qword_5013008 )
    {
      if ( *(_DWORD *)(a1 + 172) > 0x3Cu )
        v3 = sub_2CB5E90(a2, v10, v13 + 176);
    }
    else
    {
      v3 = 0;
    }
    if ( (_BYTE)qword_5012F28 && *(_DWORD *)(a1 + 172) > 0x63u )
      v3 |= sub_2CB4D10(a2, v10);
  }
  return v3;
}
