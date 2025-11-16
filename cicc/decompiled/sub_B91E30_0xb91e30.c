// Function: sub_B91E30
// Address: 0xb91e30
//
void __fastcall sub_B91E30(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rsi
  int v7; // edx
  unsigned int v8; // eax
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r15
  __int64 v12; // r12
  int v13; // edi

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v3 = sub_BD5C60(a1, a2);
    v4 = *(_QWORD *)v3;
    v5 = *(_DWORD *)(*(_QWORD *)v3 + 3248LL);
    v6 = *(_QWORD *)(v4 + 3232);
    if ( v5 )
    {
      v7 = v5 - 1;
      v8 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v9 = v6 + 40LL * v8;
      v10 = *(_QWORD *)v9;
      if ( a1 == *(_QWORD *)v9 )
      {
LABEL_4:
        v11 = *(_QWORD *)(v9 + 8);
        v12 = v11 + 16LL * *(unsigned int *)(v9 + 16);
        if ( v11 != v12 )
        {
          do
          {
            v6 = *(_QWORD *)(v12 - 8);
            v12 -= 16;
            if ( v6 )
              sub_B91220(v12 + 8, v6);
          }
          while ( v11 != v12 );
          v12 = *(_QWORD *)(v9 + 8);
        }
        if ( v12 != v9 + 24 )
          _libc_free(v12, v6);
        *(_QWORD *)v9 = -8192;
        --*(_DWORD *)(v4 + 3240);
        ++*(_DWORD *)(v4 + 3244);
      }
      else
      {
        v13 = 1;
        while ( v10 != -4096 )
        {
          v8 = v7 & (v13 + v8);
          v9 = v6 + 40LL * v8;
          v10 = *(_QWORD *)v9;
          if ( a1 == *(_QWORD *)v9 )
            goto LABEL_4;
          ++v13;
        }
      }
    }
    *(_BYTE *)(a1 + 7) &= ~0x20u;
  }
}
