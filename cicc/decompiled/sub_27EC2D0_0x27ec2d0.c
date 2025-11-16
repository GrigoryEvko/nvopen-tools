// Function: sub_27EC2D0
// Address: 0x27ec2d0
//
__int64 __fastcall sub_27EC2D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rsi
  __int64 *v5; // r11
  __int64 v6; // r10
  __int64 v7; // r9
  int v8; // ebx
  __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rax
  int v15; // edx
  int v17; // eax
  int v18; // r13d

  v4 = *(__int64 **)(a2 + 32);
  v5 = *(__int64 **)(a2 + 40);
  if ( v5 != v4 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)a3 + 72LL);
    v7 = *(unsigned int *)(*(_QWORD *)a3 + 88LL);
    v8 = v7 - 1;
    while ( 1 )
    {
      v9 = *v4;
      if ( (_DWORD)v7 )
      {
        v10 = v8 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v11 = (__int64 *)(v6 + 16LL * v10);
        v12 = *v11;
        if ( v9 == *v11 )
        {
LABEL_5:
          if ( (__int64 *)(v6 + 16 * v7) != v11 )
          {
            v13 = v11[1];
            if ( v13 )
            {
              v14 = *(_QWORD *)(v13 + 8);
              if ( v13 != v14 )
              {
                v15 = 0;
                do
                {
                  if ( !v14 )
                    BUG();
                  if ( *(_BYTE *)(v14 - 32) != 28 )
                  {
                    if ( *(_QWORD *)(v14 + 40) != a1 || v15 )
                      return 0;
                    v15 = 1;
                  }
                  v14 = *(_QWORD *)(v14 + 8);
                }
                while ( v13 != v14 );
              }
            }
          }
        }
        else
        {
          v17 = 1;
          while ( v12 != -4096 )
          {
            v18 = v17 + 1;
            v10 = v8 & (v17 + v10);
            v11 = (__int64 *)(v6 + 16LL * v10);
            v12 = *v11;
            if ( v9 == *v11 )
              goto LABEL_5;
            v17 = v18;
          }
        }
      }
      if ( v5 == ++v4 )
        return 1;
    }
  }
  return 1;
}
