// Function: sub_2E77EA0
// Address: 0x2e77ea0
//
unsigned __int64 __fastcall sub_2E77EA0(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 (*v3)(); // rax
  __int64 v5; // rax
  __int64 v6; // r13
  int v7; // ecx
  int v8; // r8d
  unsigned __int64 result; // rax
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 i; // r14
  __int64 v13; // rdx
  char *v14; // rsi
  __int64 v15; // [rsp+10h] [rbp-50h]
  int v16; // [rsp+18h] [rbp-48h]
  int v17; // [rsp+1Ch] [rbp-44h]
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v3 == sub_2DAC790 )
    BUG();
  v5 = v3();
  v6 = a2 + 320;
  v7 = *(_DWORD *)(v5 + 64);
  v8 = *(_DWORD *)(v5 + 68);
  *(_QWORD *)(a1 + 80) = 0;
  result = (unsigned __int64)v18;
  v10 = *(_QWORD *)(a2 + 328);
  if ( v10 != a2 + 320 )
  {
    do
    {
      v11 = *(_QWORD *)(v10 + 56);
      for ( i = v10 + 48; i != v11; v11 = *(_QWORD *)(v11 + 8) )
      {
        while ( 1 )
        {
          result = *(unsigned __int16 *)(v11 + 68);
          if ( v7 == (_DWORD)result || v8 == (_DWORD)result )
          {
            v13 = *(_QWORD *)(v11 + 32);
            result = *(_QWORD *)(a1 + 80);
            if ( *(_QWORD *)(v13 + 24) >= result )
              result = *(_QWORD *)(v13 + 24);
            *(_QWORD *)(a1 + 80) = result;
            if ( a3 )
            {
              v18[0] = v11;
              v14 = (char *)a3[1];
              if ( v14 == (char *)a3[2] )
              {
                v15 = v10;
                v16 = v8;
                v17 = v7;
                result = sub_2E77D20(a3, v14, v18);
                v10 = v15;
                v8 = v16;
                v7 = v17;
              }
              else
              {
                if ( v14 )
                {
                  *(_QWORD *)v14 = v11;
                  v14 = (char *)a3[1];
                }
                a3[1] = (unsigned __int64)(v14 + 8);
              }
            }
          }
          if ( (*(_BYTE *)v11 & 4) == 0 )
            break;
          v11 = *(_QWORD *)(v11 + 8);
          if ( i == v11 )
            goto LABEL_10;
        }
        while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
      }
LABEL_10:
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( v6 != v10 );
  }
  return result;
}
