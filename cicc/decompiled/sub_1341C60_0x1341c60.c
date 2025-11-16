// Function: sub_1341C60
// Address: 0x1341c60
//
unsigned __int64 __fastcall sub_1341C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  _QWORD *v5; // r15
  unsigned __int64 v6; // r13
  unsigned __int64 result; // rax
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // r13
  __int64 v10; // r12
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // rcx
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  unsigned int i; // edx
  __int64 v18; // r9
  _QWORD *v19; // rdx
  _QWORD *v20; // r9
  unsigned __int64 v21; // rax
  unsigned int v22; // [rsp+4h] [rbp-1CCh]
  __int64 v23; // [rsp+8h] [rbp-1C8h]
  __int64 v24; // [rsp+18h] [rbp-1B8h]
  _QWORD v25[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v4 = a1;
  v5 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v5 = v25;
    v22 = a4;
    v23 = a3;
    sub_130D500(v25);
    a4 = v22;
    a3 = v23;
    v4 = 0;
  }
  v6 = *(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFF000LL;
  result = *(_QWORD *)(a3 + 16) & 0xFFFFFFFFFFFFF000LL;
  v8 = v6 + result - 0x2000;
  v9 = v6 + 4096;
  v10 = (a4 << 48) | a3 & 0xFFFFFFFFFFFFLL | 1;
  if ( v8 >= v9 )
  {
    v11 = v9;
    result = 0;
    do
    {
      if ( v9 == v11 || (v11 & 0x3FFFFFFF) == 0 )
      {
        v12 = v11 & 0xFFFFFFFFC0000000LL;
        v13 = (_QWORD *)((char *)v5 + ((v11 >> 26) & 0xF0));
        v14 = *v13;
        if ( (v11 & 0xFFFFFFFFC0000000LL) == *v13 )
        {
          result = v13[1] + ((v11 >> 9) & 0x1FFFF8);
        }
        else if ( v12 == v5[32] )
        {
          v5[32] = v14;
          v15 = v5[33];
          v5[33] = v13[1];
          *v13 = v12;
          v13[1] = v15;
          result = v15 + ((v11 >> 9) & 0x1FFFF8);
        }
        else
        {
          v16 = v5 + 34;
          for ( i = 1; i != 8; ++i )
          {
            if ( v12 == *v16 )
            {
              v18 = i;
              v19 = &v5[2 * i - 2];
              v20 = &v5[2 * v18];
              v21 = v20[33];
              v20[32] = v19[32];
              v20[33] = v19[33];
              v19[32] = v14;
              v19[33] = v13[1];
              v13[1] = v21;
              *v13 = v12;
              result = ((v11 >> 9) & 0x1FFFF8) + v21;
              goto LABEL_9;
            }
            v16 += 2;
          }
          v24 = v4;
          result = sub_130D370(v4, a2, v5, v11, 1, 0);
          v4 = v24;
        }
      }
LABEL_9:
      v11 += 4096LL;
      result += 8LL;
      *(_QWORD *)(result - 8) = v10;
    }
    while ( v8 >= v11 );
  }
  return result;
}
