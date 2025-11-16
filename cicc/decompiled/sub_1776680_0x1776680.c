// Function: sub_1776680
// Address: 0x1776680
//
__int64 __fastcall sub_1776680(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r10
  __int64 v7; // rax
  int v9; // edx
  int v10; // r11d

  v2 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 56);
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 16 * v2) )
      {
        v7 = *(_QWORD *)(a1 + 80) + 16LL * *((unsigned int *)v5 + 2);
        if ( *(_QWORD *)(a1 + 88) != v7 )
          return *(_QWORD *)(v7 + 8);
      }
    }
    else
    {
      v9 = 1;
      while ( v6 != -8 )
      {
        v10 = v9 + 1;
        v4 = (v2 - 1) & (v9 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( *v5 == a2 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  return 0;
}
