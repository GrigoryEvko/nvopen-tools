// Function: sub_B23CD0
// Address: 0xb23cd0
//
void __fastcall sub_B23CD0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 *v3; // rbx
  __int64 v5; // r12
  __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 *v10; // rbx
  __int64 *v11; // r15
  unsigned int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // r12d
  __int64 v16; // rdx
  __int64 *v17; // r14
  __int64 v18; // rsi
  int v19; // eax
  int v20; // ecx
  __int64 *v21; // [rsp+0h] [rbp-A0h]
  unsigned int v24; // [rsp+24h] [rbp-7Ch]
  __int64 v25; // [rsp+28h] [rbp-78h]
  __int64 *v26[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v27[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v28; // [rsp+60h] [rbp-40h]

  if ( a1 != a2 )
  {
    v3 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *a1;
          sub_B1C5B0(v26, a3, *v3);
          v7 = *((_DWORD *)v26[2] + 2);
          sub_B1C5B0(v27, a3, v6);
          if ( v7 >= *(_DWORD *)(v28 + 8) )
            break;
          v5 = *v3;
          if ( a1 != v3 )
            memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
          ++v3;
          *a1 = v5;
          if ( a2 == v3 )
            return;
        }
        v8 = *v3;
        v9 = v3;
        v21 = v3;
        v10 = a3;
        v11 = v9;
        v25 = v8;
        v24 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
        while ( 1 )
        {
          v16 = *((unsigned int *)v10 + 6);
          v17 = v11;
          v18 = v10[1];
          if ( (_DWORD)v16 )
          {
            v12 = (v16 - 1) & v24;
            v13 = (__int64 *)(v18 + 16LL * v12);
            v14 = *v13;
            if ( v25 == *v13 )
              goto LABEL_10;
            v19 = 1;
            while ( v14 != -4096 )
            {
              v20 = v19 + 1;
              v12 = (v16 - 1) & (v19 + v12);
              v13 = (__int64 *)(v18 + 16LL * v12);
              v14 = *v13;
              if ( v25 == *v13 )
                goto LABEL_10;
              v19 = v20;
            }
          }
          v13 = (__int64 *)(v18 + 16 * v16);
LABEL_10:
          v15 = *((_DWORD *)v13 + 2);
          sub_B1C5B0(v27, v10, *--v11);
          if ( v15 >= *(_DWORD *)(v28 + 8) )
            break;
          v11[1] = *v11;
        }
        a3 = v10;
        *v17 = v25;
        v3 = v21 + 1;
      }
      while ( a2 != v21 + 1 );
    }
  }
}
