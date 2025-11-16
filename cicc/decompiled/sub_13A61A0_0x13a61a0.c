// Function: sub_13A61A0
// Address: 0x13a61a0
//
char *__fastcall sub_13A61A0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  char *result; // rax
  __int64 v6; // r12
  char *v8; // rcx
  char *v9; // r8
  char v10; // dl
  __int64 v11; // rdi

  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  result = (char *)&unk_49E97C8;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E97C8;
  *(_WORD *)(a1 + 40) = a5;
  *(_BYTE *)(a1 + 42) = a4;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 43) = 1;
  if ( a5 )
  {
    v6 = 16LL * a5;
    result = (char *)sub_2207820(v6);
    v8 = result;
    if ( result )
    {
      v9 = &result[v6];
      do
      {
        v10 = *result;
        *((_QWORD *)result + 1) = 0;
        result += 16;
        *(result - 16) = v10 & 0x80 | 0xF;
      }
      while ( result != v9 );
    }
    v11 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v8;
    if ( v11 )
      return (char *)j_j___libc_free_0_0(v11);
  }
  return result;
}
