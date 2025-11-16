// Function: sub_35AA2E0
// Address: 0x35aa2e0
//
__int64 __fastcall sub_35AA2E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 *v7; // r14
  __int64 v8; // r13
  _QWORD *v9; // rbx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208AC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208AC);
  v6 = *(__int64 **)(v5 + 240);
  v7 = *(__int64 **)(v5 + 232);
  if ( v7 != v6 )
  {
    while ( 1 )
    {
      v8 = *v7;
      v9 = sub_2EA6400(*v7);
      if ( v9 == (_QWORD *)sub_2EA64B0(v8) )
        break;
      if ( v6 == ++v7 )
        return 0;
    }
    sub_35A9700(a1, a2, v8);
  }
  return 0;
}
