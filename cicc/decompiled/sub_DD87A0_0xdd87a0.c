// Function: sub_DD87A0
// Address: 0xdd87a0
//
__int64 __fastcall sub_DD87A0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // r10
  _WORD *v18; // rdx
  __int64 v19; // r14
  _BYTE *v20; // rax
  __int64 v21; // rax
  _DWORD *v22; // rdx
  __int64 v23; // r14
  _BYTE *v24; // rax
  int v25; // r10d
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v30; // [rsp+1Ch] [rbp-44h]
  __int64 i; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 120);
  v5 = *(_QWORD *)(v4 + 32);
  result = *(_QWORD *)(v4 + 40);
  v26 = result;
  if ( result != v5 )
  {
    v27 = v5;
    v30 = a3 + 2;
    do
    {
      v7 = *(_QWORD *)(*(_QWORD *)v27 + 56LL);
      for ( i = *(_QWORD *)v27 + 48LL; i != v7; v7 = *(_QWORD *)(v7 + 8) )
      {
        if ( !v7 )
          BUG();
        if ( sub_D97040(*(_QWORD *)(a1 + 112), *(_QWORD *)(v7 - 16)) )
        {
          v8 = sub_DD8400(*(_QWORD *)(a1 + 112), v7 - 24);
          v9 = *(_QWORD *)(a1 + 8);
          v10 = (__int64)v8;
          v11 = *(unsigned int *)(a1 + 24);
          if ( (_DWORD)v11 )
          {
            v12 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v13 = (__int64 *)(v9 + 24LL * v12);
            v14 = *v13;
            if ( v10 == *v13 )
            {
LABEL_8:
              if ( v13 != (__int64 *)(v9 + 24 * v11) && v13[2] != v10 )
              {
                v15 = sub_CB69B0(a2, a3);
                v16 = *(_QWORD *)(v15 + 32);
                v17 = (_BYTE *)v15;
                if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 4 )
                {
                  v17 = (_BYTE *)sub_CB6200(v15, "[PSE]", 5u);
                }
                else
                {
                  *(_DWORD *)v16 = 1163087963;
                  *(_BYTE *)(v16 + 4) = 93;
                  *(_QWORD *)(v15 + 32) += 5LL;
                }
                v28 = (__int64)v17;
                sub_A69870(v7 - 24, v17, 0);
                v18 = *(_WORD **)(v28 + 32);
                if ( *(_QWORD *)(v28 + 24) - (_QWORD)v18 <= 1u )
                {
                  sub_CB6200(v28, (unsigned __int8 *)":\n", 2u);
                }
                else
                {
                  *v18 = 2618;
                  *(_QWORD *)(v28 + 32) += 2LL;
                }
                v19 = sub_CB69B0(a2, v30);
                sub_D955C0(v10, v19);
                v20 = *(_BYTE **)(v19 + 32);
                if ( *(_BYTE **)(v19 + 24) == v20 )
                {
                  sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
                }
                else
                {
                  *v20 = 10;
                  ++*(_QWORD *)(v19 + 32);
                }
                v21 = sub_CB69B0(a2, v30);
                v22 = *(_DWORD **)(v21 + 32);
                v23 = v21;
                if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 3u )
                {
                  v23 = sub_CB6200(v21, "--> ", 4u);
                }
                else
                {
                  *v22 = 540945709;
                  *(_QWORD *)(v21 + 32) += 4LL;
                }
                sub_D955C0(v13[2], v23);
                v24 = *(_BYTE **)(v23 + 32);
                if ( *(_BYTE **)(v23 + 24) == v24 )
                {
                  sub_CB6200(v23, (unsigned __int8 *)"\n", 1u);
                }
                else
                {
                  *v24 = 10;
                  ++*(_QWORD *)(v23 + 32);
                }
              }
            }
            else
            {
              v25 = 1;
              while ( v14 != -4096 )
              {
                v12 = (v11 - 1) & (v25 + v12);
                v13 = (__int64 *)(v9 + 24LL * v12);
                v14 = *v13;
                if ( v10 == *v13 )
                  goto LABEL_8;
                ++v25;
              }
            }
          }
        }
      }
      v27 += 8;
      result = v27;
    }
    while ( v26 != v27 );
  }
  return result;
}
