// Function: sub_228CD10
// Address: 0x228cd10
//
void __fastcall sub_228CD10(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  unsigned __int64 v5; // r12
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int64 v9; // r8
  char v10; // dl
  unsigned __int64 v11; // rdi

  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A08E50;
  *(_WORD *)(a1 + 40) = a5;
  *(_BYTE *)(a1 + 42) = a4;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 43) = 1;
  if ( a5 )
  {
    v5 = 16LL * a5;
    v7 = sub_2207820(v5);
    v8 = v7;
    if ( v7 )
    {
      v9 = v5 + v7;
      do
      {
        v10 = *(_BYTE *)v7;
        *(_QWORD *)(v7 + 8) = 0;
        v7 += 16;
        *(_BYTE *)(v7 - 16) = v10 & 0x80 | 0xF;
      }
      while ( v7 != v9 );
    }
    v11 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v8;
    if ( v11 )
      j_j___libc_free_0_0(v11);
  }
}
