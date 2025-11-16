// Function: sub_BA6130
// Address: 0xba6130
//
void __fastcall sub_BA6130(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // r8
  unsigned int v5; // esi
  __int64 *v6; // rax
  __int64 v7; // r10
  __int64 v8; // r12
  int v9; // eax
  int v10; // r11d

  v1 = **(__int64 ***)(a1 + 8);
  v2 = *v1;
  v3 = *(unsigned int *)(*v1 + 592);
  v4 = *(_QWORD *)(*v1 + 576);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a1 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
      {
        *v6 = -8192;
        v8 = v6[1];
        --*(_DWORD *)(v2 + 584);
        ++*(_DWORD *)(v2 + 588);
        sub_BA6110((const __m128i *)(v8 + 8), 0);
        if ( v8 )
        {
          if ( (*(_BYTE *)(v8 + 32) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v8 + 40), 24LL * *(unsigned int *)(v8 + 48), 8);
          j_j___libc_free_0(v8, 144);
        }
      }
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v10 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a1 == *v6 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
}
