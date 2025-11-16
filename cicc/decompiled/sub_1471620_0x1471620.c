// Function: sub_1471620
// Address: 0x1471620
//
__int64 __fastcall sub_1471620(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // r9d
  unsigned int v12; // edx
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  _WORD *v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  _DWORD *v21; // rdx
  __int64 v22; // r8
  _BYTE *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v30; // [rsp+1Ch] [rbp-44h]
  __int64 i; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 120);
  v5 = *(_QWORD *)(v4 + 32);
  result = *(_QWORD *)(v4 + 40);
  v24 = result;
  if ( result != v5 )
  {
    v25 = v5;
    v30 = a3 + 2;
    do
    {
      v7 = *(_QWORD *)(*(_QWORD *)v25 + 48LL);
      for ( i = *(_QWORD *)v25 + 40LL; i != v7; v7 = *(_QWORD *)(v7 + 8) )
      {
        if ( !v7 )
          BUG();
        if ( sub_1456C80(*(_QWORD *)(a1 + 112), *(_QWORD *)(v7 - 24)) )
        {
          v8 = sub_146F1B0(*(_QWORD *)(a1 + 112), v7 - 24);
          v9 = *(unsigned int *)(a1 + 24);
          if ( (_DWORD)v9 )
          {
            v10 = *(_QWORD *)(a1 + 8);
            v11 = 1;
            v12 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v13 = (__int64 *)(v10 + 24LL * v12);
            v14 = *v13;
            if ( v8 == *v13 )
            {
LABEL_8:
              if ( v13 != (__int64 *)(v10 + 24 * v9) && v13[2] != v8 )
              {
                v15 = sub_16E8750(a2, a3);
                v16 = *(_QWORD *)(v15 + 24);
                v17 = v15;
                if ( (unsigned __int64)(*(_QWORD *)(v15 + 16) - v16) <= 4 )
                {
                  v17 = sub_16E7EE0(v15, "[PSE]", 5);
                }
                else
                {
                  *(_DWORD *)v16 = 1163087963;
                  *(_BYTE *)(v16 + 4) = 93;
                  *(_QWORD *)(v15 + 24) += 5LL;
                }
                v26 = v17;
                sub_155C2B0(v7 - 24, v17, 0);
                v18 = *(_WORD **)(v26 + 24);
                if ( *(_QWORD *)(v26 + 16) - (_QWORD)v18 <= 1u )
                {
                  sub_16E7EE0(v26, ":\n", 2);
                }
                else
                {
                  *v18 = 2618;
                  *(_QWORD *)(v26 + 24) += 2LL;
                }
                v27 = sub_16E8750(a2, v30);
                sub_1456620(v8, v27);
                v19 = *(_BYTE **)(v27 + 24);
                if ( *(_BYTE **)(v27 + 16) == v19 )
                {
                  sub_16E7EE0(v27, "\n", 1);
                }
                else
                {
                  *v19 = 10;
                  ++*(_QWORD *)(v27 + 24);
                }
                v20 = sub_16E8750(a2, v30);
                v21 = *(_DWORD **)(v20 + 24);
                v22 = v20;
                if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 3u )
                {
                  v22 = sub_16E7EE0(v20, "--> ", 4);
                }
                else
                {
                  *v21 = 540945709;
                  *(_QWORD *)(v20 + 24) += 4LL;
                }
                v28 = v22;
                sub_1456620(v13[2], v22);
                v23 = *(_BYTE **)(v28 + 24);
                if ( *(_BYTE **)(v28 + 16) == v23 )
                {
                  sub_16E7EE0(v28, "\n", 1);
                }
                else
                {
                  *v23 = 10;
                  ++*(_QWORD *)(v28 + 24);
                }
              }
            }
            else
            {
              while ( v14 != -8 )
              {
                v12 = (v9 - 1) & (v11 + v12);
                v13 = (__int64 *)(v10 + 24LL * v12);
                v14 = *v13;
                if ( v8 == *v13 )
                  goto LABEL_8;
                ++v11;
              }
            }
          }
        }
      }
      v25 += 8;
      result = v25;
    }
    while ( v24 != v25 );
  }
  return result;
}
