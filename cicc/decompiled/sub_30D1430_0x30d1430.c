// Function: sub_30D1430
// Address: 0x30d1430
//
void __fastcall sub_30D1430(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rax
  __int64 v6; // r10
  int v7; // eax
  int v8; // r11d

  v2 = *(unsigned int *)(a1 + 792);
  v3 = *(_QWORD *)(a1 + 776);
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
        *(_DWORD *)(a1 + 652) += *((_DWORD *)v5 + 2);
        *(_DWORD *)(a1 + 748) -= *((_DWORD *)v5 + 2);
        *v5 = -8192;
        --*(_DWORD *)(a1 + 784);
        ++*(_DWORD *)(a1 + 788);
      }
    }
    else
    {
      v7 = 1;
      while ( v6 != -4096 )
      {
        v8 = v7 + 1;
        v4 = (v2 - 1) & (v7 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v7 = v8;
      }
    }
  }
}
