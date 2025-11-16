// Function: sub_64A300
// Address: 0x64a300
//
char __fastcall sub_64A300(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 i; // r12
  __int64 v10; // r13
  __int64 v11; // r15

  v3 = *(_QWORD *)(a1 + 264);
  v4 = *(_QWORD *)(a1 + 152);
  if ( v3 )
    *(_QWORD *)(a1 + 264) = 0;
  else
    v3 = a2;
  v5 = sub_64A1A0(v4, v3);
  v8 = v4;
  for ( i = v5; *(_BYTE *)(v8 + 140) == 12; v8 = *(_QWORD *)(v8 + 160) )
    ;
  for ( ; *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  v10 = *(_QWORD *)(v5 + 168);
  if ( !*(_QWORD *)v10 && (*(_BYTE *)(a1 + 207) & 0x10) == 0 )
  {
    v11 = *(_QWORD *)(v8 + 168);
    if ( v4 == i || (LODWORD(v5) = sub_8D97D0(i, v4, 0, v6, v7), (_DWORD)v5) )
    {
      LOBYTE(v5) = *(_QWORD *)(v10 + 56) == 0;
      if ( (*(_QWORD *)(v11 + 56) == 0) == (_BYTE)v5 )
      {
        LOBYTE(v5) = *(_BYTE *)(v10 + 16) ^ *(_BYTE *)(v11 + 16);
        if ( (v5 & 8) == 0 && *(_BYTE *)(v4 + 140) == 7 )
          i = v4;
      }
    }
  }
  *(_QWORD *)(a1 + 264) = i;
  return v5;
}
