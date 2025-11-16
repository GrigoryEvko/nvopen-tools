// Function: sub_278A710
// Address: 0x278a710
//
__int64 __fastcall sub_278A710(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned int v6; // edi
  __int64 *v7; // rax
  __int64 v8; // rdx
  int v10; // eax
  int v11; // r11d

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v3 )
  {
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( a3 || v7 != (__int64 *)(16 * v3 + v4) )
        return *((unsigned int *)v7 + 2);
      return 0;
    }
    v10 = 1;
    while ( v8 != -4096 )
    {
      v11 = v10 + 1;
      v6 = (v3 - 1) & (v10 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v10 = v11;
    }
  }
  if ( !a3 )
    return 0;
  return *(unsigned int *)(v4 + 16LL * (unsigned int)v3 + 8);
}
