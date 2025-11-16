// Function: sub_D23E00
// Address: 0xd23e00
//
__int64 __fastcall sub_D23E00(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rax
  __int64 v6; // r10
  int v8; // eax
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 72);
  v3 = *(_QWORD *)(a1 + 56);
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
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * *((int *)v5 + 2)) = 0;
        *v5 = -8192;
        --*(_DWORD *)(a1 + 64);
        ++*(_DWORD *)(a1 + 68);
        return 1;
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
  return 0;
}
