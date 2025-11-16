// Function: sub_3247D90
// Address: 0x3247d90
//
bool __fastcall sub_3247D90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r9
  int v8; // edx
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 368);
  v3 = *(_QWORD *)(a1 + 352);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 16 * v2) )
        return 88LL * *((unsigned int *)v5 + 2) != 88LL * *(unsigned int *)(a1 + 384);
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
        if ( *v5 == a2 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return 0;
}
