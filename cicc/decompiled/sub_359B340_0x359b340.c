// Function: sub_359B340
// Address: 0x359b340
//
__int64 __fastcall sub_359B340(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // rcx
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  _BYTE *v10; // r13
  _BYTE *v11; // r15
  _BYTE *v12; // rbx
  __int64 v13; // rax
  char v14; // al
  _BYTE *v15; // r15
  __int64 v16; // rbx
  __int64 result; // rax
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  char v27; // [rsp+38h] [rbp-38h]

  v5 = *(_QWORD *)a3;
  v24 = *(_QWORD *)a3;
  v25 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v25 )
  {
    do
    {
      v6 = *(_QWORD *)(v25 - 8);
      v7 = *(_QWORD *)(v6 + 48);
      v8 = v6 + 48;
      v9 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 != v8 )
      {
        do
        {
          while ( 1 )
          {
            if ( (unsigned int)*(unsigned __int16 *)(v9 + 68) - 1 > 1 )
            {
              LOBYTE(v26) = 0;
              if ( (unsigned __int8)sub_2E8B400(v9, (__int64)&v26, a3, v5, a5)
                || !*(_WORD *)(v9 + 68)
                || *(_WORD *)(v9 + 68) == 68 )
              {
                a3 = *(_QWORD *)(v9 + 32);
                v10 = (_BYTE *)(a3 + 40LL * (*(_DWORD *)(v9 + 40) & 0xFFFFFF));
                if ( (_BYTE *)a3 != v10 )
                {
                  v11 = *(_BYTE **)(v9 + 32);
                  while ( 1 )
                  {
                    v12 = v11;
                    if ( sub_2DADC00(v11) )
                      break;
                    v11 += 40;
                    if ( v10 == v11 )
                      goto LABEL_4;
                  }
                  if ( v11 != v10 )
                    break;
                }
              }
            }
LABEL_4:
            v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v8 == v9 )
              goto LABEL_22;
          }
          do
          {
            v13 = *((unsigned int *)v12 + 2);
            if ( (unsigned int)(v13 - 1) > 0x3FFFFFFE )
            {
              v22 = a1[3];
              if ( (int)v13 < 0 )
                v23 = *(_QWORD *)(*(_QWORD *)(v22 + 56) + 16 * (v13 & 0x7FFFFFFF) + 8);
              else
                v23 = *(_QWORD *)(*(_QWORD *)(v22 + 304) + 8 * v13);
              if ( v23 )
              {
                if ( (*(_BYTE *)(v23 + 3) & 0x10) != 0 )
                {
                  while ( 1 )
                  {
                    v23 = *(_QWORD *)(v23 + 32);
                    if ( !v23 )
                      break;
                    if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 )
                      goto LABEL_39;
                  }
                }
                else
                {
LABEL_39:
                  v5 = a1[6];
LABEL_40:
                  a3 = *(_QWORD *)(v23 + 16);
                  if ( v5 != *(_QWORD *)(a3 + 24) )
                    goto LABEL_4;
                  while ( 1 )
                  {
                    v23 = *(_QWORD *)(v23 + 32);
                    if ( !v23 )
                      break;
                    if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 )
                      goto LABEL_40;
                  }
                }
              }
            }
            else
            {
              a3 = (unsigned __int8)v12[3];
              v14 = (unsigned __int8)a3 >> 4;
              LOBYTE(a3) = (unsigned __int8)a3 >> 6;
              if ( (v14 & 1 & (unsigned __int8)a3) == 0 )
                goto LABEL_4;
            }
            if ( v12 + 40 == v10 )
              break;
            v15 = v12 + 40;
            while ( 1 )
            {
              v12 = v15;
              if ( sub_2DADC00(v15) )
                break;
              v15 += 40;
              if ( v10 == v15 )
                goto LABEL_21;
            }
          }
          while ( v10 != v15 );
LABEL_21:
          sub_2FAD510(*(_QWORD *)(a1[5] + 32LL), v9);
          v16 = *(_QWORD *)v9;
          sub_2E88E20(v9);
          v9 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        }
        while ( v8 != (v16 & 0xFFFFFFFFFFFFFFF8LL) );
      }
LABEL_22:
      v25 -= 8;
    }
    while ( v24 != v25 );
  }
  result = sub_2E311E0(a2);
  v18 = *(_QWORD *)(a2 + 56);
  v19 = result;
  v26 = v18;
  if ( v18 != result )
  {
    while ( 1 )
    {
      v27 = 1;
      sub_2FD79B0(&v26);
      v20 = *(unsigned int *)(*(_QWORD *)(v18 + 32) + 8LL);
      v21 = a1[3];
      if ( (int)v20 >= 0 )
      {
        result = *(_QWORD *)(*(_QWORD *)(v21 + 304) + 8 * v20);
        if ( result )
          goto LABEL_26;
LABEL_30:
        sub_2FAD510(*(_QWORD *)(a1[5] + 32LL), v18);
        result = sub_2E88E20(v18);
        v18 = v26;
        if ( v26 == v19 )
          return result;
      }
      else
      {
        result = *(_QWORD *)(*(_QWORD *)(v21 + 56) + 16 * (v20 & 0x7FFFFFFF) + 8);
        if ( !result )
          goto LABEL_30;
LABEL_26:
        while ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
        {
          result = *(_QWORD *)(result + 32);
          if ( !result )
            goto LABEL_30;
        }
        v18 = v26;
        if ( v26 == v19 )
          return result;
      }
    }
  }
  return result;
}
