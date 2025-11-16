// Function: sub_30D3D40
// Address: 0x30d3d40
//
void __fastcall sub_30D3D40(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  int v8; // eax
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 816);
  v3 = *(_QWORD *)(a1 + 800);
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
        v7 = *(int *)(a1 + 716) + (__int64)*((int *)v5 + 2);
        if ( v7 > 0x7FFFFFFF )
          v7 = 0x7FFFFFFF;
        if ( v7 < (__int64)0xFFFFFFFF80000000LL )
          LODWORD(v7) = 0x80000000;
        *(_DWORD *)(a1 + 716) = v7;
        *(_DWORD *)(a1 + 780) -= *((_DWORD *)v5 + 2);
        *(_DWORD *)(a1 + 784) += *((_DWORD *)v5 + 2);
        *v5 = -8192;
        --*(_DWORD *)(a1 + 808);
        ++*(_DWORD *)(a1 + 812);
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
