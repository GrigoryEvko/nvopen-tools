// Function: sub_F7DFE0
// Address: 0xf7dfe0
//
_BYTE *__fastcall sub_F7DFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v6; // ax
  __int64 v7; // rax
  __int64 v8; // rdx
  _BYTE **v9; // r12
  _BYTE **v10; // r14
  _BYTE *v11; // r15
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // r10
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  int v27; // eax

  if ( *(_BYTE *)(a1 + 512) || !(unsigned __int8)sub_DB2B50(*(_QWORD *)a1, a2) )
  {
    v6 = *(_WORD *)(a2 + 24);
    if ( v6 )
    {
      if ( v6 != 15 )
      {
        v7 = sub_D97100(*(_QWORD *)a1, a2);
        v9 = (_BYTE **)(v7 + 8 * v8);
        v10 = (_BYTE **)v7;
        if ( (_BYTE **)v7 != v9 )
        {
          do
          {
            while ( 1 )
            {
              v11 = *v10;
              if ( **v10 <= 0x1Cu
                || *((_QWORD *)v11 + 1) != sub_D95540(a2)
                || !(unsigned __int8)sub_B19DB0(*(_QWORD *)(*(_QWORD *)a1 + 40LL), (__int64)v11, a3) )
              {
                goto LABEL_6;
              }
              v13 = *(_QWORD *)a1;
              v14 = *((_QWORD *)v11 + 5);
              v15 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
              v16 = *(_DWORD *)(v15 + 24);
              v17 = *(_QWORD *)(v15 + 8);
              if ( v16 )
              {
                v18 = v16 - 1;
                v19 = v18 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
                v20 = (__int64 *)(v17 + 16LL * v19);
                v21 = *v20;
                if ( v14 != *v20 )
                {
                  v27 = 1;
                  while ( v21 != -4096 )
                  {
                    v12 = (unsigned int)(v27 + 1);
                    v19 = v18 & (v27 + v19);
                    v20 = (__int64 *)(v17 + 16LL * v19);
                    v21 = *v20;
                    if ( v14 == *v20 )
                      goto LABEL_12;
                    v27 = v12;
                  }
                  goto LABEL_18;
                }
LABEL_12:
                v22 = v20[1];
                if ( v22 )
                  break;
              }
LABEL_18:
              if ( (unsigned __int8)sub_D9BB00(v13, a2, v11, a4, v12) )
                return v11;
              ++v10;
              *(_DWORD *)(a4 + 8) = 0;
              if ( v9 == v10 )
                return 0;
            }
            v23 = *(_QWORD *)(a3 + 40);
            if ( *(_BYTE *)(v22 + 84) )
            {
              v24 = *(_QWORD **)(v22 + 64);
              v25 = &v24[*(unsigned int *)(v22 + 76)];
              if ( v24 == v25 )
                goto LABEL_6;
              while ( v23 != *v24 )
              {
                if ( v25 == ++v24 )
                  goto LABEL_6;
              }
              goto LABEL_18;
            }
            if ( sub_C8CA60(v22 + 56, v23) )
            {
              v13 = *(_QWORD *)a1;
              goto LABEL_18;
            }
LABEL_6:
            ++v10;
          }
          while ( v9 != v10 );
        }
      }
    }
  }
  return 0;
}
