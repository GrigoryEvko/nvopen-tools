// Function: sub_35D6260
// Address: 0x35d6260
//
__int64 __fastcall sub_35D6260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 *v22; // [rsp+18h] [rbp-48h]
  __int64 v23[7]; // [rsp+28h] [rbp-38h] BYREF

  result = *(_QWORD *)(**(_QWORD **)(a1 + 16) + 2216LL);
  if ( (__int64 (*)())result != sub_302E1B0 )
  {
    result = ((__int64 (*)(void))result)();
    if ( (_BYTE)result )
    {
      result = *(unsigned int *)(a1 + 144);
      if ( (_DWORD)result )
      {
        if ( a5 != a3 )
        {
          while ( 1 )
          {
            if ( !a3 )
              BUG();
            result = *(unsigned __int8 *)(a3 - 24);
            v7 = a3 - 24;
            if ( (unsigned __int8)(result - 34) > 0x33u )
            {
              if ( (_BYTE)result == 30 )
              {
                v23[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 16) + 72LL) + 120LL);
                result = sub_A74390(v23, 74, 0);
                if ( (_BYTE)result )
                  result = sub_35D5F90(a1, a3 - 24, a2, *(_QWORD *)(a1 + 128));
              }
              goto LABEL_11;
            }
            v8 = 0x8000000000041LL;
            if ( _bittest64(&v8, (unsigned int)(result - 34)) )
            {
              if ( (_DWORD)result == 40 )
              {
                v9 = -32 - 32LL * (unsigned int)sub_B491D0(a3 - 24);
              }
              else
              {
                v9 = -32;
                if ( (_DWORD)result != 85 )
                {
                  if ( (_DWORD)result != 34 )
                    BUG();
                  v9 = -96;
                }
              }
              if ( *(char *)(a3 - 17) < 0 )
              {
                v10 = sub_BD2BC0(a3 - 24);
                v12 = v10 + v11;
                v13 = v10 + v11;
                if ( *(char *)(a3 - 17) >= 0 )
                {
                  if ( (unsigned int)(v13 >> 4) )
LABEL_45:
                    BUG();
                }
                else if ( (unsigned int)((v12 - sub_BD2BC0(a3 - 24)) >> 4) )
                {
                  if ( *(char *)(a3 - 17) >= 0 )
                    goto LABEL_45;
                  v14 = *(_DWORD *)(sub_BD2BC0(a3 - 24) + 8);
                  if ( *(char *)(a3 - 17) >= 0 )
                    BUG();
                  v15 = sub_BD2BC0(a3 - 24);
                  v9 -= 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
                }
              }
              v22 = (__int64 *)(v7 + v9);
              result = 32LL * (*(_DWORD *)(a3 - 20) & 0x7FFFFFF);
              if ( v7 - result != v7 + v9 )
              {
                v17 = (__int64 *)(v7 - result);
                v18 = 0;
                do
                {
                  result = sub_BD6020(*v17);
                  if ( (_BYTE)result )
                  {
                    v18 = *v17;
                    result = sub_35D5F90(a1, a3 - 24, a2, *v17);
                  }
                  v17 += 4;
                }
                while ( v22 != v17 );
                if ( v18 )
                  goto LABEL_30;
              }
LABEL_11:
              a3 = *(_QWORD *)(a3 + 8);
              if ( a5 == a3 )
                return result;
            }
            else if ( (_BYTE)result == 61 )
            {
              v19 = *(_QWORD *)(a3 - 56);
              result = sub_BD6020(v19);
              if ( !(_BYTE)result )
                goto LABEL_11;
              result = sub_35D5F90(a1, a3 - 24, a2, v19);
              a3 = *(_QWORD *)(a3 + 8);
              if ( a5 == a3 )
                return result;
            }
            else
            {
              if ( (_BYTE)result != 62 )
                goto LABEL_11;
              v18 = *(_QWORD *)(a3 - 56);
              result = sub_BD6020(v18);
              if ( (_BYTE)result )
              {
LABEL_30:
                result = sub_35D5BC0(a1, a3 - 24, a2, v18);
                a3 = *(_QWORD *)(a3 + 8);
                if ( a5 == a3 )
                  return result;
              }
              else
              {
                a3 = *(_QWORD *)(a3 + 8);
                if ( a5 == a3 )
                  return result;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
