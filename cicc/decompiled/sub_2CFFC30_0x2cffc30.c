// Function: sub_2CFFC30
// Address: 0x2cffc30
//
__int64 __fastcall sub_2CFFC30(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_12;
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_12;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v7 = (__int64 *)a1[1];
  v8 = v6 + 176;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F8FBD4 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_12;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8FBD4);
  a1[22] = v8;
  v12 = (__int64)(a1 + 22);
  a1[23] = v11 + 176;
  if ( (_DWORD)qword_5014A48 )
    return sub_2CFEB90(v12, a2);
  else
    return 0;
}
