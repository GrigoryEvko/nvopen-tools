// Function: sub_3247E30
// Address: 0x3247e30
//
void __fastcall sub_3247E30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r9
  unsigned int v6; // r8d
  __int64 *v7; // rax
  __int64 v8; // r11
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rcx
  int v16; // eax
  int v17; // ebx

  v4 = *(unsigned int *)(a1 + 368);
  v5 = *(_QWORD *)(a1 + 352);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a3 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        v9 = *(_QWORD *)(a1 + 376);
        v10 = v9 + 88LL * *((unsigned int *)v7 + 2);
        if ( v10 != v9 + 88LL * *(unsigned int *)(a1 + 384) )
        {
          v11 = *(unsigned int *)(v10 + 16);
          if ( (_DWORD)v11 )
          {
            v12 = 8 * v11;
            v13 = 0;
            do
            {
              v14 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + v13);
              *(_QWORD *)(v14 + 40) = a2 & 0xFFFFFFFFFFFFFFFBLL;
              v15 = *(_QWORD **)(a2 + 32);
              if ( v15 )
              {
                *(_QWORD *)v14 = *v15;
                **(_QWORD **)(a2 + 32) = v14 & 0xFFFFFFFFFFFFFFFBLL;
              }
              v13 += 8;
              *(_QWORD *)(a2 + 32) = v14;
            }
            while ( v12 != v13 );
            *(_DWORD *)(v10 + 16) = 0;
          }
        }
      }
    }
    else
    {
      v16 = 1;
      while ( v8 != -4096 )
      {
        v17 = v16 + 1;
        v6 = (v4 - 1) & (v16 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a3 )
          goto LABEL_3;
        v16 = v17;
      }
    }
  }
}
