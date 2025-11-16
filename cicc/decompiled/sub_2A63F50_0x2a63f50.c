// Function: sub_2A63F50
// Address: 0x2a63f50
//
__int64 __fastcall sub_2A63F50(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // rdi
  int v11; // eax
  int v12; // ecx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  int v17; // edx
  int v18; // eax
  int v19; // r9d
  int v20; // r11d

  v2 = *a1;
  v3 = *(_QWORD *)(*a1 + 2544);
  v4 = *(unsigned int *)(v2 + 2560);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( v5 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v3 + 16 * v4) )
      {
        v9 = v7[1];
        v10 = *(_QWORD *)(v9 + 32);
        v11 = *(_DWORD *)(v9 + 48);
        if ( v11 )
        {
          v12 = v11 - 1;
          v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v14 = (__int64 *)(v10 + 16LL * v13);
          v15 = *v14;
          if ( a2 == *v14 )
            return v14[1];
          v18 = 1;
          while ( v15 != -4096 )
          {
            v19 = v18 + 1;
            v13 = v12 & (v18 + v13);
            v14 = (__int64 *)(v10 + 16LL * v13);
            v15 = *v14;
            if ( a2 == *v14 )
              return v14[1];
            v18 = v19;
          }
        }
      }
    }
    else
    {
      v17 = 1;
      while ( v8 != -4096 )
      {
        v20 = v17 + 1;
        v6 = (v4 - 1) & (v17 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( v5 == *v7 )
          goto LABEL_3;
        v17 = v20;
      }
    }
  }
  return 0;
}
