// Function: sub_2B48560
// Address: 0x2b48560
//
void __fastcall sub_2B48560(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  char **v12; // r8
  _QWORD *v13; // rbx
  char **v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // r8
  int v17; // eax
  _QWORD *v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v3 = 0x1C71C71C71C71C7LL;
  *a1 = a3;
  a1[1] = 0;
  if ( a3 <= 0x1C71C71C71C71C7LL )
    v3 = a3;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v20 = (__int64)a2;
      v6 = sub_2207800(72 * v3);
      a2 = (_QWORD *)v20;
      v7 = (_QWORD *)v6;
      if ( v6 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v8 = (_QWORD *)(v6 + 72 * v3);
    v9 = *(_QWORD *)v20;
    v10 = v20 + 8;
    v7[1] = v7 + 3;
    v11 = *(unsigned int *)(v20 + 16);
    v12 = (char **)(v7 + 1);
    *v7 = v9;
    v7[2] = 0xC00000000LL;
    if ( (_DWORD)v11 )
    {
      sub_2B0D090((__int64)(v7 + 1), (char **)(v20 + 8), v11, v20, (__int64)v12, v10);
      v9 = *v7;
      a2 = (_QWORD *)v20;
      v10 = v20 + 8;
      v12 = (char **)(v7 + 1);
    }
    v13 = v7 + 9;
    if ( v8 != v7 + 9 )
    {
      do
      {
        while ( 1 )
        {
          *v13 = v9;
          v16 = (__int64)(v13 - 9);
          v13[1] = v13 + 3;
          v17 = *((_DWORD *)v13 - 14);
          *((_DWORD *)v13 + 4) = 0;
          *((_DWORD *)v13 + 5) = 12;
          if ( !v17 )
            break;
          v14 = (char **)(v13 - 8);
          v15 = (__int64)(v13 + 1);
          v13 += 9;
          v18 = a2;
          v19 = v16;
          v21 = v10;
          sub_2B0D090(v15, v14, v11, (__int64)a2, v16, v10);
          a2 = v18;
          v16 = v19;
          v10 = v21;
          v9 = *(v13 - 9);
          if ( v8 == v13 )
            goto LABEL_13;
        }
        v13 += 9;
        v9 = *(v13 - 9);
      }
      while ( v8 != v13 );
LABEL_13:
      v12 = (char **)(v16 + 80);
    }
    *a2 = v9;
    sub_2B0D090(v10, v12, v11, (__int64)a2, (__int64)v12, v10);
    a1[2] = (__int64)v7;
    a1[1] = v3;
  }
}
