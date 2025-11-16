// Function: sub_37E70D0
// Address: 0x37e70d0
//
__int64 __fastcall sub_37E70D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 (*v7)(); // rdx
  __int64 v8; // rax

  v2 = a2 + 48;
  v3 = *(_QWORD *)(a2 + 56);
  if ( v3 == a2 + 48 )
    return 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(a1 + 520);
      v7 = *(__int64 (**)())(*(_QWORD *)v6 + 168LL);
      v8 = 0xFFFFFFFFLL;
      if ( v7 != sub_2E77FD0 )
        v8 = ((unsigned int (__fastcall *)(__int64, __int64, __int64 (*)(), __int64 (*)()))v7)(v6, v3, v7, sub_2E77FD0);
      v5 += v8;
      if ( !v3 )
        BUG();
      if ( (*(_BYTE *)v3 & 4) == 0 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        return v5;
    }
    while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
      v3 = *(_QWORD *)(v3 + 8);
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v2 != v3 );
  return v5;
}
