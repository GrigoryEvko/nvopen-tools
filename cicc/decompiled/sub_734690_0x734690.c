// Function: sub_734690
// Address: 0x734690
//
__int64 __fastcall sub_734690(_QWORD *a1)
{
  __int64 v2; // rbx
  _QWORD *i; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rsi

  v2 = a1[4];
  sub_86AAE0();
  if ( dword_4D047EC )
  {
    for ( i = (_QWORD *)a1[26]; i; i = (_QWORD *)*i )
      *(_BYTE *)(i[1] + 169LL) &= ~0x10u;
  }
  v4 = *(int *)(v2 + 160);
  *(_BYTE *)(v2 + 193) &= ~0x20u;
  *(_BYTE *)(v2 + 202) &= ~0x80u;
  *(_QWORD *)(unk_4F072B8 + 16 * v4) = 0;
  *(_DWORD *)(unk_4F072B8 + 16LL * *(int *)(v2 + 160) + 8) = 0;
  v5 = *(_QWORD *)(v2 + 152);
  *(_BYTE *)(v2 + 194) &= ~0x20u;
  for ( *(_DWORD *)(v2 + 160) = 0; *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  *(_QWORD *)(*(_QWORD *)(v5 + 168) + 8LL) = 0;
  if ( !*(_BYTE *)(v2 + 172) )
  {
    *(_BYTE *)(v2 + 205) &= ~2u;
    *(_BYTE *)(v2 + 172) = 1;
  }
  result = 0xFFFFFFFEFDFFFFFFLL;
  v7 = *(unsigned int *)(v2 + 164);
  *(_QWORD *)(v2 + 200) &= 0xFFFFFFFEFDFFFFFFLL;
  if ( (_DWORD)v7 )
  {
    v8 = (_QWORD *)a1[1];
    v9 = *a1;
    if ( v8 )
      *v8 = v9;
    else
      *(_QWORD *)(unk_4F072B0 + 8LL * (int)v7) = v9;
    if ( *a1 )
      *(_QWORD *)(*a1 + 8LL) = a1[1];
    result = unk_4F072B0;
    if ( !*(_QWORD *)(unk_4F072B0 + 8LL * (int)v7) )
      result = sub_823310(v7);
  }
  *(_DWORD *)(v2 + 164) = 0;
  return result;
}
