// Function: sub_2FAD510
// Address: 0x2fad510
//
void __fastcall sub_2FAD510(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rax
  __int64 v6; // r10
  __int64 v7; // rdx
  int v8; // eax
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 144);
  v3 = *(_QWORD *)(a1 + 128);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 16 * v2) )
      {
        v7 = v5[1];
        *v5 = -8192;
        --*(_DWORD *)(a1 + 136);
        ++*(_DWORD *)(a1 + 140);
        *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 16) = 0;
      }
    }
    else
    {
      v8 = 1;
      while ( v6 != -4096 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
}
