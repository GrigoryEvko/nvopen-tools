// Function: sub_2E11710
// Address: 0x2e11710
//
void __fastcall sub_2E11710(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD *v9; // rsi
  unsigned __int16 *v10; // rax
  unsigned __int16 v11; // r14
  __int64 v12; // rdi
  unsigned int v13; // eax
  int v14; // r11d
  __int16 *v15; // rbx
  char v16; // r12
  int v17; // eax
  unsigned __int16 *v18; // rax
  unsigned __int16 v19; // bx
  int v20; // r14d
  __int16 *v21; // r12
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int16 v24; // [rsp+14h] [rbp-3Ch]
  char v25; // [rsp+17h] [rbp-39h]
  int v26; // [rsp+18h] [rbp-38h]
  unsigned int v27; // [rsp+1Ch] [rbp-34h]

  v5 = a3;
  sub_2E1DCC0(a1[6], *a1, a1[4], a1[5], a1 + 7);
  v9 = (_QWORD *)a1[2];
  v23 = 4 * v5;
  v10 = (unsigned __int16 *)(v9[6] + 4 * v5);
  v11 = *v10;
  v24 = v10[1];
  if ( *v10 )
  {
    v25 = 0;
    while ( 1 )
    {
      v12 = v11;
      v13 = v11;
      v14 = v11;
      v6 = v9[7];
      v8 = v6 + 2LL * *(unsigned int *)(v9[1] + 24LL * v11 + 8);
      if ( v8 )
      {
        v6 = a1[1];
        v15 = (__int16 *)v8;
        v16 = 1;
        while ( 1 )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8 * v12) )
          {
            v26 = v14;
            v27 = v13;
            sub_3506EE0(a1[6], a2, v13);
            v6 = a1[1];
            v14 = v26;
            v13 = v27;
          }
          v9 = *(_QWORD **)(v6 + 384);
          v7 = v9[v13 >> 6] & (1LL << v11);
          if ( !v7 )
            v16 = 0;
          v17 = *v15++;
          if ( !(_WORD)v17 )
            break;
          v14 += v17;
          v13 = (unsigned __int16)v14;
          LOBYTE(v11) = v14;
          v12 = (unsigned __int16)v14;
        }
        v25 |= v16;
      }
      else
      {
        v25 = 1;
      }
      v11 = v24;
      if ( !v24 )
        break;
      v9 = (_QWORD *)a1[2];
      v24 = 0;
    }
    if ( !v25 )
    {
      v9 = (_QWORD *)a1[2];
      v18 = (unsigned __int16 *)(v9[6] + v23);
      v6 = *v18;
      v19 = v18[1];
      if ( (_WORD)v6 )
      {
        while ( 1 )
        {
          v7 = (unsigned __int16)v6;
          v20 = (unsigned __int16)v6;
          v21 = (__int16 *)(v9[7] + 2LL * *(unsigned int *)(v9[1] + 24LL * (unsigned __int16)v6 + 8));
          if ( v21 )
          {
            while ( 1 )
            {
              if ( *(_QWORD *)(*(_QWORD *)(a1[1] + 304LL) + 8 * v7) )
              {
                v9 = (_QWORD *)a2;
                sub_3507070(a1[6], a2, v6, -1, -1, 0);
              }
              v22 = *v21++;
              if ( !(_WORD)v22 )
                break;
              v20 += v22;
              v6 = (unsigned __int16)v20;
              v7 = (unsigned __int16)v20;
            }
          }
          if ( !v19 )
            break;
          v6 = v19;
          v9 = (_QWORD *)a1[2];
          v19 = 0;
        }
      }
    }
  }
  if ( LOBYTE(qword_501EA48[8]) )
    sub_2E0AC50(a2, (__int64)v9, v6, v7, v8);
}
