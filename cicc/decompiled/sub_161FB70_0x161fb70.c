// Function: sub_161FB70
// Address: 0x161fb70
//
void __fastcall sub_161FB70(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  int v4; // eax
  int v5; // edx
  int v6; // edi
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 *v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  __int64 v13; // rsi

  if ( (*(_BYTE *)(a1 + 34) & 0x10) != 0 )
  {
    v2 = sub_16498A0(a1);
    v3 = *(_QWORD *)v2;
    v4 = *(_DWORD *)(*(_QWORD *)v2 + 2760LL);
    if ( v4 )
    {
      v5 = v4 - 1;
      v6 = 1;
      v7 = *(_QWORD *)(v3 + 2744);
      v8 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v9 = (__int64 *)(v7 + 40LL * v8);
      v10 = *v9;
      if ( a1 == *v9 )
      {
LABEL_4:
        v11 = v9[1];
        v12 = v11 + 16LL * *((unsigned int *)v9 + 4);
        if ( v11 != v12 )
        {
          do
          {
            v13 = *(_QWORD *)(v12 - 8);
            v12 -= 16LL;
            if ( v13 )
              sub_161E7C0(v12 + 8, v13);
          }
          while ( v11 != v12 );
          v12 = v9[1];
        }
        if ( (__int64 *)v12 != v9 + 3 )
          _libc_free(v12);
        *v9 = -16;
        --*(_DWORD *)(v3 + 2752);
        ++*(_DWORD *)(v3 + 2756);
      }
      else
      {
        while ( v10 != -8 )
        {
          v8 = v5 & (v6 + v8);
          v9 = (__int64 *)(v7 + 40LL * v8);
          v10 = *v9;
          if ( a1 == *v9 )
            goto LABEL_4;
          ++v6;
        }
      }
    }
    *(_DWORD *)(a1 + 32) &= ~0x100000u;
  }
}
