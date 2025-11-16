// Function: sub_630B30
// Address: 0x630b30
//
__int64 __fastcall sub_630B30(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rax

  v2 = a2;
  v4 = 1;
  v8 = sub_726BB0(1);
  if ( dword_4F077C4 == 2 )
  {
    v4 = a2;
    if ( (unsigned int)sub_8D23B0(a2) )
    {
      v4 = a2;
      if ( (unsigned int)sub_8D3A70(a2) )
      {
        a2 = 0;
        v4 = v2;
        sub_8AD220(v2, 0);
      }
    }
  }
  v9 = sub_725160(v4, a2, v5, v6, v7);
  *(_QWORD *)(v8 + 16) = v9;
  *(_QWORD *)(v9 + 40) = v2;
  *(_BYTE *)(*(_QWORD *)(v8 + 16) + 96LL) |= 1u;
  if ( *(_QWORD *)(a1 + 16) )
    **(_QWORD **)(a1 + 24) = v8;
  else
    *(_QWORD *)(a1 + 16) = v8;
  *(_QWORD *)(a1 + 24) = v8;
  return v8;
}
