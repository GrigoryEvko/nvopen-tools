// Function: sub_1A7A5C0
// Address: 0x1a7a5c0
//
__int64 __fastcall sub_1A7A5C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // r13

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
  v3 = *(_QWORD *)(v2 + 80);
  v4 = v2 + 72;
  if ( v3 != v4 )
  {
    v5 = 0;
    while ( 1 )
    {
      v6 = v3 - 24;
      if ( !v3 )
        v6 = 0;
      v7 = sub_157EBA0(v6);
      if ( *(_BYTE *)(v7 + 16) == 25 && a1 != v7 )
      {
        v8 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
        if ( !sub_1A7A490(v8, a2, v7) || v5 && v8 != v5 )
          return 0;
        v5 = v8;
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        return v5;
    }
  }
  return 0;
}
