// Function: sub_29AABA0
// Address: 0x29aaba0
//
__int64 __fastcall sub_29AABA0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  int v5; // ebx
  unsigned int v6; // r14d
  int v7; // ecx
  unsigned int v8; // edx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // edx
  __int64 *v13; // rcx
  int v15; // r8d

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    return 0;
  if ( !v2 )
    BUG();
  v3 = v2 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 <= 0xA )
  {
    v5 = sub_B46E30(v3);
    if ( v5 )
    {
      v6 = 0;
      while ( 1 )
      {
        v10 = sub_B46EC0(v3, v6);
        v11 = *(_QWORD *)(*a1 + 8LL);
        v12 = *(_DWORD *)(*a1 + 24LL);
        if ( v12 )
        {
          v7 = v12 - 1;
          v8 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v9 = *(_QWORD *)(v11 + 8LL * v8);
          if ( v10 == v9 )
            goto LABEL_7;
          v15 = 1;
          while ( v9 != -4096 )
          {
            v8 = v7 & (v15 + v8);
            v9 = *(_QWORD *)(v11 + 8LL * v8);
            if ( v10 == v9 )
              goto LABEL_7;
            ++v15;
          }
        }
        v13 = (__int64 *)a1[1];
        if ( *v13 )
        {
          if ( *v13 != v10 )
            return 1;
LABEL_7:
          if ( v5 == ++v6 )
            return 0;
        }
        else
        {
          ++v6;
          *v13 = v10;
          if ( v5 == v6 )
            return 0;
        }
      }
    }
  }
  return 0;
}
