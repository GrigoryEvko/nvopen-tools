// Function: sub_317AE00
// Address: 0x317ae00
//
unsigned __int64 __fastcall sub_317AE00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdi
  signed __int64 v14; // rax
  int v15; // ecx
  int v16; // edx
  bool v17; // of
  int v19; // edi
  int v20; // edx
  int v23; // [rsp+1Ch] [rbp-34h]

  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 )
  {
    v23 = 0;
    v4 = 0;
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 24);
      if ( *(_BYTE *)v6 > 0x1Cu )
      {
        v7 = *(_QWORD *)(v6 + 40);
        if ( (unsigned __int8)sub_2A64220(*(__int64 **)(a1 + 56), v7) )
        {
          v10 = *(unsigned int *)(a1 + 120);
          v11 = *(_QWORD *)(a1 + 104);
          if ( !(_DWORD)v10 )
            goto LABEL_9;
          v12 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v13 = (__int64 *)(v11 + 8LL * v12);
          v9 = *v13;
          if ( v7 != *v13 )
          {
            v19 = 1;
            while ( v9 != -4096 )
            {
              v20 = v19 + 1;
              v12 = (v10 - 1) & (v19 + v12);
              v13 = (__int64 *)(v11 + 8LL * v12);
              v9 = *v13;
              if ( v7 == *v13 )
                goto LABEL_8;
              v19 = v20;
            }
LABEL_9:
            v14 = sub_317A680(a1, v6, a2, a3, v8, v9);
            v15 = 1;
            if ( v16 != 1 )
              v15 = v23;
            v17 = __OFADD__(v14, v4);
            v4 += v14;
            v23 = v15;
            if ( v17 )
            {
              v4 = 0x8000000000000000LL;
              if ( v14 > 0 )
                v4 = 0x7FFFFFFFFFFFFFFFLL;
            }
            goto LABEL_3;
          }
LABEL_8:
          if ( v13 == (__int64 *)(v11 + 8 * v10) )
            goto LABEL_9;
        }
      }
LABEL_3:
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return v4;
    }
  }
  return 0;
}
