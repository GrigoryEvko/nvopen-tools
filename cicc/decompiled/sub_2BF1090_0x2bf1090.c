// Function: sub_2BF1090
// Address: 0x2bf1090
//
void __fastcall sub_2BF1090(
        __int64 a1,
        __int64 a2,
        unsigned __int8 (__fastcall *a3)(__int64, __int64, _QWORD),
        __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  char v8; // r15
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // r8
  __int64 *v13; // r8
  __int64 v14; // r15
  _QWORD *v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // r9
  int v20; // r11d
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // r14
  const void *v24; // [rsp+8h] [rbp-78h]
  unsigned __int8 v25; // [rsp+1Bh] [rbp-65h]
  char v26; // [rsp+1Bh] [rbp-65h]
  unsigned int v27; // [rsp+1Ch] [rbp-64h]
  __int64 v30; // [rsp+30h] [rbp-50h]
  __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a1 != a2 )
  {
    v27 = 0;
    v4 = a1;
    v5 = a2;
    v24 = (const void *)(a2 + 32);
    while ( v27 < *(_DWORD *)(v4 + 24) )
    {
      v6 = 0;
      v30 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL * v27) + 16LL);
      if ( (_DWORD)v30 )
      {
        v7 = v5;
        v31 = v4;
        v8 = 0;
        v9 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL * v27);
        v10 = v7;
        do
        {
          while ( 1 )
          {
            v11 = 8 * v6;
            if ( v31 == *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8 * v6) )
            {
              if ( a3(a4, v9, (unsigned int)v6) )
                break;
            }
            if ( v30 == ++v6 )
              goto LABEL_15;
          }
          v12 = *(_QWORD *)(v9 + 8);
          v32[0] = v9;
          v13 = (__int64 *)(v11 + v12);
          v14 = *v13;
          v15 = *(_QWORD **)(*v13 + 16);
          v16 = (__int64)&v15[*(unsigned int *)(*v13 + 24)];
          v17 = sub_2BEF3B0(v15, v16, v32);
          if ( (_QWORD *)v16 != v17 )
          {
            if ( (_QWORD *)v16 != v17 + 1 )
            {
              v25 = v19;
              memmove(v17, v17 + 1, v16 - (_QWORD)(v17 + 1));
              v20 = *(_DWORD *)(v14 + 24);
              v19 = v25;
            }
            *(_DWORD *)(v14 + 24) = v20 - 1;
            v18 = (_QWORD *)(v11 + *(_QWORD *)(v9 + 8));
          }
          *v18 = v10;
          v21 = *(unsigned int *)(v10 + 24);
          if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
          {
            v26 = v19;
            sub_C8D5F0(v10 + 16, v24, v21 + 1, 8u, (__int64)v18, v19);
            v21 = *(unsigned int *)(v10 + 24);
            LOBYTE(v19) = v26;
          }
          v8 = v19;
          ++v6;
          *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v21) = v9;
          ++*(_DWORD *)(v10 + 24);
        }
        while ( v30 != v6 );
LABEL_15:
        v22 = v10;
        v23 = v8;
        v4 = v31;
        v5 = v22;
        if ( v23 )
          continue;
      }
      ++v27;
    }
  }
}
