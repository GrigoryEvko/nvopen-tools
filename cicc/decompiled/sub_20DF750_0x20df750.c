// Function: sub_20DF750
// Address: 0x20df750
//
__int64 __fastcall sub_20DF750(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 (*v7)(); // rdx
  __int64 v8; // rax

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  if ( v3 == a2 + 24 )
    return 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(a1 + 472);
      v7 = *(__int64 (**)())(*(_QWORD *)v6 + 128LL);
      v8 = 0xFFFFFFFFLL;
      if ( v7 != sub_1F39410 )
        v8 = ((unsigned int (__fastcall *)(__int64, __int64, __int64 (*)(), __int64 (*)()))v7)(v6, v3, v7, sub_1F39410);
      v5 += v8;
      if ( !v3 )
        BUG();
      if ( (*(_BYTE *)v3 & 4) == 0 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        return v5;
    }
    while ( (*(_BYTE *)(v3 + 46) & 8) != 0 )
      v3 = *(_QWORD *)(v3 + 8);
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v2 != v3 );
  return v5;
}
