// Function: sub_B9A9D0
// Address: 0xb9a9d0
//
void __fastcall sub_B9A9D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r8
  unsigned int v6; // eax
  unsigned int **v7; // rdi
  unsigned int *v8; // rcx
  int v9; // edi
  int v10; // r10d

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v3 = *(_QWORD *)sub_BD5C60(a1, a2);
    v4 = *(unsigned int *)(v3 + 3248);
    v5 = *(_QWORD *)(v3 + 3232);
    if ( (_DWORD)v4 )
    {
      v6 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v7 = (unsigned int **)(v5 + 40LL * v6);
      v8 = *v7;
      if ( (unsigned int *)a1 == *v7 )
      {
LABEL_4:
        sub_B9A820(v7 + 1, a2);
        return;
      }
      v9 = 1;
      while ( v8 != (unsigned int *)-4096LL )
      {
        v10 = v9 + 1;
        v6 = (v4 - 1) & (v9 + v6);
        v7 = (unsigned int **)(v5 + 40LL * v6);
        v8 = *v7;
        if ( (unsigned int *)a1 == *v7 )
          goto LABEL_4;
        v9 = v10;
      }
    }
    v7 = (unsigned int **)(v5 + 40 * v4);
    goto LABEL_4;
  }
}
