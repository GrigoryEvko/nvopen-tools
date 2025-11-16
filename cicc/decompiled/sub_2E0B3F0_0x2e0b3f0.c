// Function: sub_2E0B3F0
// Address: 0x2e0b3f0
//
unsigned __int64 __fastcall sub_2E0B3F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rsi
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // r13
  unsigned int v13; // r12d
  __int64 v14; // rdi
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _DWORD *v17; // rdx
  _QWORD v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)a1;
    v5 = *(_QWORD *)a1 + 24 * v3;
    do
    {
      v6 = v4;
      v4 += 24;
      sub_2E0B2D0(a2, v6);
    }
    while ( v5 != v4 );
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 4 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"EMPTY", 5u);
    }
    else
    {
      *(_DWORD *)v8 = 1414548805;
      *(_BYTE *)(v8 + 4) = 89;
      *(_QWORD *)(a2 + 32) += 5LL;
    }
  }
  result = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)result )
  {
    v9 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 32);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v9 + 1;
      *v9 = 32;
    }
    v10 = *(__int64 **)(a1 + 64);
    result = *(unsigned int *)(a1 + 72);
    v11 = &v10[result];
    if ( v10 != v11 )
    {
      v12 = *v10;
      v13 = 0;
      while ( 1 )
      {
        v14 = sub_CB59D0(a2, v13);
        v15 = *(_BYTE **)(v14 + 32);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
        {
          sub_CB5D20(v14, 64);
        }
        else
        {
          *(_QWORD *)(v14 + 32) = v15 + 1;
          *v15 = 64;
        }
        if ( (*(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v18[0] = *(_QWORD *)(v12 + 8);
          result = sub_2FAD600(v18, a2);
          if ( (*(_BYTE *)(v12 + 8) & 6) == 0 )
          {
            v17 = *(_DWORD **)(a2 + 32);
            result = *(_QWORD *)(a2 + 24) - (_QWORD)v17;
            if ( result <= 3 )
            {
              result = sub_CB6200(a2, (unsigned __int8 *)"-phi", 4u);
            }
            else
            {
              *v17 = 1768452141;
              *(_QWORD *)(a2 + 32) += 4LL;
            }
          }
        }
        else
        {
          result = *(_QWORD *)(a2 + 32);
          if ( result >= *(_QWORD *)(a2 + 24) )
          {
            result = sub_CB5D20(a2, 120);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = result + 1;
            *(_BYTE *)result = 120;
          }
        }
        ++v10;
        ++v13;
        if ( v11 == v10 )
          break;
        v12 = *v10;
        if ( v13 )
        {
          v16 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 32);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v16 + 1;
            *v16 = 32;
          }
        }
      }
    }
  }
  return result;
}
