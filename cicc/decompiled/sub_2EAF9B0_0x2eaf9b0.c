// Function: sub_2EAF9B0
// Address: 0x2eaf9b0
//
__int64 __fastcall sub_2EAF9B0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  v3 = sub_B2BE50(*a2);
  if ( (unsigned __int8)sub_B6E900(v3) )
  {
    v8 = *(__int64 **)(a1 + 8);
    v9 = *v8;
    v10 = v8[1];
    if ( v9 == v10 )
LABEL_13:
      BUG();
    while ( *(_UNKNOWN **)v9 != &unk_503BDA8 )
    {
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_13;
    }
    v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_503BDA8);
    v4 = sub_3503E60(v11);
  }
  else
  {
    v4 = 0;
  }
  v5 = (_QWORD *)sub_22077B0(0x10u);
  if ( v5 )
  {
    *v5 = a2;
    v5[1] = v4;
  }
  v6 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 200) = v5;
  if ( v6 )
    j_j___libc_free_0(v6);
  return 0;
}
