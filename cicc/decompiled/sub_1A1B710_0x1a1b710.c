// Function: sub_1A1B710
// Address: 0x1a1b710
//
void __fastcall sub_1A1B710(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rbx
  const void *v5; // r12
  unsigned int v6; // r8d
  int v7; // r9d
  char v8; // dl
  _QWORD *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  _QWORD *v12; // rcx
  __int64 v13; // rax

  v2 = *(_QWORD *)(a2 + 8);
  if ( v2 )
  {
    v4 = a1 + 96;
    v5 = (const void *)(a1 + 32);
    do
    {
      while ( 1 )
      {
        v9 = sub_1648700(v2);
        v10 = *(_QWORD **)(a1 + 104);
        if ( *(_QWORD **)(a1 + 112) != v10 )
          break;
        v11 = &v10[*(unsigned int *)(a1 + 124)];
        v6 = *(_DWORD *)(a1 + 124);
        if ( v10 != v11 )
        {
          v12 = 0;
          while ( v9 != (_QWORD *)*v10 )
          {
            if ( *v10 == -2 )
              v12 = v10;
            if ( v11 == ++v10 )
            {
              if ( !v12 )
                goto LABEL_17;
              *v12 = v9;
              --*(_DWORD *)(a1 + 128);
              ++*(_QWORD *)(a1 + 96);
              goto LABEL_14;
            }
          }
          goto LABEL_4;
        }
LABEL_17:
        if ( v6 >= *(_DWORD *)(a1 + 120) )
          break;
        *(_DWORD *)(a1 + 124) = ++v6;
        *v11 = v9;
        v13 = *(unsigned int *)(a1 + 24);
        ++*(_QWORD *)(a1 + 96);
        if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 28) )
        {
LABEL_19:
          sub_16CD150(a1 + 16, v5, 0, 8, v6, v7);
          v13 = *(unsigned int *)(a1 + 24);
        }
LABEL_15:
        *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v13) = v2;
        ++*(_DWORD *)(a1 + 24);
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return;
      }
      sub_16CCBA0(v4, (__int64)v9);
      if ( v8 )
      {
LABEL_14:
        v13 = *(unsigned int *)(a1 + 24);
        if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 28) )
          goto LABEL_19;
        goto LABEL_15;
      }
LABEL_4:
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v2 );
  }
}
