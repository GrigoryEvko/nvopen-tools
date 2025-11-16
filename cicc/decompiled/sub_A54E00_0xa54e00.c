// Function: sub_A54E00
// Address: 0xa54e00
//
__int64 __fastcall sub_A54E00(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v10; // rax

  *(_QWORD *)a1 = &unk_49DC840;
  if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 16) )
    sub_CB5AE0(a1);
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    if ( !*(_DWORD *)(a1 + 44) || v4 )
    {
      v5 = *(_QWORD *)(a1 + 24) - v4;
    }
    else
    {
      v10 = sub_CB54F0(a1);
      v3 = *(_QWORD *)(a1 + 48);
      v5 = v10;
    }
    v6 = *(_QWORD *)(v3 + 32);
    v7 = *(_QWORD *)(v3 + 16);
    if ( v5 )
    {
      if ( v6 != v7 )
        sub_CB5AE0(v3);
      a2 = sub_2207820(v5);
      sub_CB5980(v3, a2, v5, 1);
    }
    else
    {
      if ( v6 != v7 )
        sub_CB5AE0(v3);
      a2 = 0;
      sub_CB5980(v3, 0, 0, 0);
    }
  }
  v8 = *(_QWORD *)(a1 + 72);
  if ( v8 != a1 + 96 )
    _libc_free(v8, a2);
  sub_CB5840(a1);
  return j_j___libc_free_0(a1, 112);
}
