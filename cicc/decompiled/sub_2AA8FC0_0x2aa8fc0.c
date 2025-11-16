// Function: sub_2AA8FC0
// Address: 0x2aa8fc0
//
__int64 __fastcall sub_2AA8FC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r10
  __int64 v7; // rcx
  int v9; // edx
  int v10; // r11d

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v2 )
  {
LABEL_7:
    v7 = *(_QWORD *)(a1 + 32);
    return v7 + 184LL * *(unsigned int *)(a1 + 40);
  }
  v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v3 + 16LL * v4);
  v6 = *v5;
  if ( a2 != *v5 )
  {
    v9 = 1;
    while ( v6 != -4096 )
    {
      v10 = v9 + 1;
      v4 = (v2 - 1) & (v9 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        goto LABEL_3;
      v9 = v10;
    }
    goto LABEL_7;
  }
LABEL_3:
  v7 = *(_QWORD *)(a1 + 32);
  if ( v5 != (__int64 *)(v3 + 16 * v2) )
    return v7 + 184LL * *((unsigned int *)v5 + 2);
  return v7 + 184LL * *(unsigned int *)(a1 + 40);
}
