// Function: sub_2732E30
// Address: 0x2732e30
//
void __fastcall sub_2732E30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rcx
  char **v5; // rax
  __int64 v6; // r9
  char **v7; // r12
  char **v8; // r13
  char **v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // r8
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v3 = 0xC30C30C30C30C3LL;
  *a1 = a3;
  a1[1] = 0;
  if ( a3 <= 0xC30C30C30C30C3LL )
    v3 = a3;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    v4 = (__int64)a1;
    while ( 1 )
    {
      v19 = v4;
      v5 = (char **)sub_2207800(168 * v3);
      v4 = v19;
      v7 = v5;
      if ( v5 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v8 = &v5[21 * v3];
    v9 = v5;
    v10 = *(unsigned int *)(a2 + 8);
    *v5 = (char *)(v5 + 2);
    v5[1] = (char *)0x800000000LL;
    if ( (_DWORD)v10 )
    {
      sub_272D8A0((__int64)v5, (char **)a2, v10, v19, (__int64)v5, v6);
      v4 = v19;
      v9 = v7;
    }
    v11 = (__int64)(v7 + 21);
    v7[18] = *(char **)(a2 + 144);
    v7[19] = *(char **)(a2 + 152);
    *((_DWORD *)v7 + 40) = *(_DWORD *)(a2 + 160);
    if ( v8 == v7 + 21 )
    {
      v12 = (__int64)v7;
    }
    else
    {
      do
      {
        *(_DWORD *)(v11 + 8) = 0;
        v12 = v11;
        v14 = v11 - 168;
        *(_QWORD *)v11 = v11 + 16;
        v15 = *(_DWORD *)(v11 - 160);
        *(_DWORD *)(v11 + 12) = 8;
        if ( v15 )
        {
          v17 = v4;
          sub_272D8A0(v11, (char **)(v11 - 168), v10, v4, v14, v11);
          v4 = v17;
          v12 = v11;
          v14 = v11 - 168;
        }
        v13 = *(_QWORD *)(v11 - 24);
        v11 += 168;
        *(_QWORD *)(v11 - 24) = v13;
        *(_QWORD *)(v11 - 16) = *(_QWORD *)(v11 - 184);
        *(_DWORD *)(v11 - 8) = *(_DWORD *)(v11 - 176);
      }
      while ( v8 != (char **)v11 );
      v9 = (char **)(v14 + 168);
    }
    v18 = v4;
    v20 = v12;
    sub_272D8A0(a2, v9, v10, v4, (__int64)v9, v12);
    v16 = *(_QWORD *)(v20 + 144);
    *(_QWORD *)(v18 + 16) = v7;
    *(_QWORD *)(v18 + 8) = v3;
    *(_QWORD *)(a2 + 144) = v16;
    *(_QWORD *)(a2 + 152) = *(_QWORD *)(v20 + 152);
    *(_DWORD *)(a2 + 160) = *(_DWORD *)(v20 + 160);
  }
}
