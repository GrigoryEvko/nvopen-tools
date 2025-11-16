// Function: sub_1F32E10
// Address: 0x1f32e10
//
__int64 __fastcall sub_1F32E10(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 *v5; // rdx
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v2 = sub_1636880(a1, *a2);
  if ( (_BYTE)v2 )
  {
    return 0;
  }
  else
  {
    v5 = *(__int64 **)(a1 + 8);
    v6 = v2;
    v7 = *v5;
    v8 = v5[1];
    if ( v7 == v8 )
LABEL_11:
      BUG();
    while ( *(_UNKNOWN **)v7 != &unk_4FC5828 )
    {
      v7 += 16;
      if ( v8 == v7 )
        goto LABEL_11;
    }
    v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4FC5828);
    sub_1F33BD0(a1 + 232, a2, *(unsigned __int8 *)(a1 + 400), v9, 0, 0);
    do
    {
      v3 = v6;
      v6 = sub_1F392E0(a1 + 232);
    }
    while ( (_BYTE)v6 );
  }
  return v3;
}
