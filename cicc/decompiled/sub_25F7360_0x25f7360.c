// Function: sub_25F7360
// Address: 0x25f7360
//
__int64 __fastcall sub_25F7360(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r10
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 *v16; // r9
  __int64 v17; // r11
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // r9d
  int v24; // [rsp+4h] [rbp-3Ch]
  __int64 v25; // [rsp+8h] [rbp-38h]

  result = sub_AA5930(a1);
  v25 = v8;
  if ( v8 != result )
  {
    v9 = result;
    while ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) == 0 )
    {
LABEL_23:
      result = *(_QWORD *)(v9 + 32);
      if ( !result )
        BUG();
      v9 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v9 = result - 24;
      if ( v25 == v9 )
        return result;
    }
    v10 = 0;
    v11 = 8LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
    while ( 1 )
    {
      v12 = *(unsigned int *)(a4 + 24);
      v13 = *(_QWORD *)(a4 + 8);
      v14 = *(_QWORD *)(*(_QWORD *)(v9 - 8) + 32LL * *(unsigned int *)(v9 + 72) + v10);
      if ( (_DWORD)v12 )
      {
        v15 = (v12 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v16 = (__int64 *)(v13 + 8LL * v15);
        v17 = *v16;
        if ( v14 == *v16 )
        {
LABEL_7:
          if ( v16 != (__int64 *)(v13 + 8 * v12) )
          {
            v18 = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v18 == v14 + 48 )
              goto LABEL_34;
            if ( !v18 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
LABEL_34:
              BUG();
            if ( *(_BYTE *)(v18 - 24) != 31 )
              BUG();
            v19 = (_QWORD *)(v18 - 56);
            v20 = v18 - 88 - 32LL * ((*(_DWORD *)(v18 - 20) & 0x7FFFFFF) == 3);
            do
            {
              while ( !*v19 )
              {
                if ( a2 )
                  goto LABEL_13;
LABEL_18:
                *v19 = a3;
                if ( !a3 )
                  goto LABEL_13;
                v22 = *(_QWORD *)(a3 + 16);
                v19[1] = v22;
                if ( v22 )
                  *(_QWORD *)(v22 + 16) = v19 + 1;
                v19[2] = a3 + 16;
                *(_QWORD *)(a3 + 16) = v19;
                v19 -= 4;
                if ( (_QWORD *)v20 == v19 )
                  goto LABEL_22;
              }
              if ( a2 == *v19 )
              {
                v21 = v19[1];
                *(_QWORD *)v19[2] = v21;
                if ( v21 )
                  *(_QWORD *)(v21 + 16) = v19[2];
                goto LABEL_18;
              }
LABEL_13:
              v19 -= 4;
            }
            while ( (_QWORD *)v20 != v19 );
          }
        }
        else
        {
          v23 = 1;
          while ( v17 != -4096 )
          {
            v15 = (v12 - 1) & (v23 + v15);
            v24 = v23 + 1;
            v16 = (__int64 *)(v13 + 8LL * v15);
            v17 = *v16;
            if ( v14 == *v16 )
              goto LABEL_7;
            v23 = v24;
          }
        }
      }
LABEL_22:
      v10 += 8;
      if ( v11 == v10 )
        goto LABEL_23;
    }
  }
  return result;
}
