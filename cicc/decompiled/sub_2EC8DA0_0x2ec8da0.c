// Function: sub_2EC8DA0
// Address: 0x2ec8da0
//
__int64 __fastcall sub_2EC8DA0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  _DWORD *v3; // rdx
  int v5; // esi
  unsigned int v6; // eax
  unsigned int v7; // ecx
  unsigned int v8; // ecx
  unsigned int v9; // eax
  __int64 *v10; // rdi
  __int64 v11; // rax
  bool v12; // cf
  int v13; // eax
  int v14; // edx
  __int64 result; // rax
  __int64 v16; // rax
  void (*v17)(); // rax
  int v18; // eax

  v2 = a2;
  v3 = *(_DWORD **)(a1 + 8);
  if ( !v3[1] && a2 < *(_DWORD *)(a1 + 172) )
    v2 = *(_DWORD *)(a1 + 172);
  v5 = *(_DWORD *)(a1 + 164);
  v6 = (v2 - v5) * *v3;
  v7 = *(_DWORD *)(a1 + 168) - v6;
  if ( *(_DWORD *)(a1 + 168) <= v6 )
    v7 = 0;
  *(_DWORD *)(a1 + 168) = v7;
  v8 = *(_DWORD *)(a1 + 180);
  v9 = v5 + v8 - v2;
  if ( v2 - v5 > v8 )
    v9 = 0;
  v10 = *(__int64 **)(a1 + 152);
  *(_DWORD *)(a1 + 180) = v9;
  if ( !*((_DWORD *)v10 + 2) )
  {
    *(_DWORD *)(a1 + 164) = v2;
    goto LABEL_10;
  }
  if ( v5 != v2 )
  {
    v16 = *v10;
    if ( *(_DWORD *)(a1 + 24) == 1 )
      goto LABEL_20;
    while ( 1 )
    {
      v17 = *(void (**)())(v16 + 88);
      if ( v17 != nullsub_1621 )
        goto LABEL_21;
      while ( 1 )
      {
        v18 = *(_DWORD *)(a1 + 164) + 1;
        *(_DWORD *)(a1 + 164) = v18;
        if ( v18 == v2 )
        {
          v3 = *(_DWORD **)(a1 + 8);
          goto LABEL_10;
        }
        v16 = **(_QWORD **)(a1 + 152);
        if ( *(_DWORD *)(a1 + 24) != 1 )
          break;
LABEL_20:
        v17 = *(void (**)())(v16 + 80);
        if ( v17 != nullsub_1620 )
LABEL_21:
          v17();
      }
    }
  }
LABEL_10:
  v11 = *(unsigned int *)(a1 + 276);
  v12 = *(_DWORD *)(a1 + 176) < v2;
  *(_BYTE *)(a1 + 160) = 1;
  if ( !v12 )
    v2 = *(_DWORD *)(a1 + 176);
  if ( (_DWORD)v11 )
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v11);
  else
    v13 = v3[72] * *(_DWORD *)(a1 + 184);
  v14 = v3[73];
  result = v13 - v14 * v2;
  *(_BYTE *)(a1 + 280) = (int)result >= v14;
  return result;
}
