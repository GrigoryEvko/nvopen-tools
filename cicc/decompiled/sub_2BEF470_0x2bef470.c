// Function: sub_2BEF470
// Address: 0x2bef470
//
bool __fastcall sub_2BEF470(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r10
  unsigned int v7; // ecx
  __int64 *v8; // r8
  __int64 v9; // r11
  bool result; // al
  int v11; // r8d
  int v12; // r12d

  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v4 )
    return 0;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 56LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v11 = 1;
    while ( v9 != -4096 )
    {
      v12 = v11 + 1;
      v7 = (v4 - 1) & (v11 + v7);
      v8 = (__int64 *)(v5 + 56LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v11 = v12;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v5 + 56 * v4) )
    return 0;
  if ( a4 == 1 )
    a3 += *(_DWORD *)(a1 + 8);
  result = 0;
  if ( a3 < *((_DWORD *)v8 + 4) )
    return *(_QWORD *)(v8[1] + 8LL * a3) != 0;
  return result;
}
