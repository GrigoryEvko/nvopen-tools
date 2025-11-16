// Function: sub_210E250
// Address: 0x210e250
//
__int64 __fastcall sub_210E250(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r9
  unsigned int v4; // r8d
  __int64 *v5; // rax
  __int64 v6; // rdx
  int v8; // eax
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 160);
  v3 = *(_QWORD *)(a1 + 144);
  if ( !(_DWORD)v2 )
  {
LABEL_6:
    v5 = (__int64 *)(v3 + 16 * v2);
    return *(_QWORD *)(a1 + 168) + 24LL * *((unsigned int *)v5 + 2);
  }
  v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v3 + 16LL * v4);
  v6 = *v5;
  if ( a2 != *v5 )
  {
    v8 = 1;
    while ( v6 != -8 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        return *(_QWORD *)(a1 + 168) + 24LL * *((unsigned int *)v5 + 2);
      v8 = v9;
    }
    goto LABEL_6;
  }
  return *(_QWORD *)(a1 + 168) + 24LL * *((unsigned int *)v5 + 2);
}
