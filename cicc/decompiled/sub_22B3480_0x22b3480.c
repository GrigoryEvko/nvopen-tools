// Function: sub_22B3480
// Address: 0x22b3480
//
void __fastcall sub_22B3480(__int64 *a1, _DWORD *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  int v9; // eax
  char **v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rbx
  char **v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // r8
  int v17; // eax
  _DWORD *v18; // [rsp+8h] [rbp-48h]
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
      a2 = (_DWORD *)v20;
      v7 = v6;
      if ( v6 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v8 = v6 + 72 * v3;
    v9 = *(_DWORD *)v20;
    v10 = (char **)(v7 + 8);
    *(_QWORD *)(v7 + 8) = v7 + 24;
    v11 = *(unsigned int *)(v20 + 16);
    v12 = v20 + 8;
    *(_DWORD *)v7 = v9;
    *(_QWORD *)(v7 + 16) = 0xC00000000LL;
    if ( (_DWORD)v11 )
    {
      sub_22AD4A0(v7 + 8, (char **)(v20 + 8), v11, v20, (__int64)v10, v12);
      v9 = *(_DWORD *)v7;
      a2 = (_DWORD *)v20;
      v12 = v20 + 8;
      v10 = (char **)(v7 + 8);
    }
    v13 = v7 + 72;
    if ( v8 != v7 + 72 )
    {
      do
      {
        while ( 1 )
        {
          *(_DWORD *)v13 = v9;
          v16 = v13 - 72;
          *(_QWORD *)(v13 + 8) = v13 + 24;
          v17 = *(_DWORD *)(v13 - 56);
          *(_DWORD *)(v13 + 16) = 0;
          *(_DWORD *)(v13 + 20) = 12;
          if ( !v17 )
            break;
          v14 = (char **)(v13 - 64);
          v15 = v13 + 8;
          v13 += 72;
          v18 = a2;
          v19 = v16;
          v21 = v12;
          sub_22AD4A0(v15, v14, v11, (__int64)a2, v16, v12);
          a2 = v18;
          v16 = v19;
          v12 = v21;
          v9 = *(_DWORD *)(v13 - 72);
          if ( v8 == v13 )
            goto LABEL_13;
        }
        v13 += 72;
        v9 = *(_DWORD *)(v13 - 72);
      }
      while ( v8 != v13 );
LABEL_13:
      v10 = (char **)(v16 + 80);
    }
    *a2 = v9;
    sub_22AD4A0(v12, v10, v11, (__int64)a2, (__int64)v10, v12);
    a1[2] = v7;
    a1[1] = v3;
  }
}
