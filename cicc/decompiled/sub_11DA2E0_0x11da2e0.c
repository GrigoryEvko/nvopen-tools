// Function: sub_11DA2E0
// Address: 0x11da2e0
//
unsigned __int64 __fastcall sub_11DA2E0(__int64 a1, unsigned int *a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int *v4; // r13
  unsigned __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  int v9; // r14d
  unsigned __int64 v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  unsigned int *v17; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  unsigned __int64 v20; // [rsp+28h] [rbp-48h]
  int v21[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v4 = a2;
  result = sub_B491C0(a1);
  v19 = result;
  if ( result )
  {
    result = (unsigned __int64)&a2[a3];
    v17 = (unsigned int *)result;
    if ( a2 != (unsigned int *)result )
    {
      do
      {
        while ( 1 )
        {
          v7 = *v4;
          v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (v7 - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 8LL);
          if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
            v8 = **(_QWORD **)(v8 + 16);
          v9 = *(_DWORD *)(v8 + 8) >> 8;
          if ( !sub_B2F070(v19, v9) || (unsigned __int8)sub_B49B80(a1, v7, 43) )
          {
            v10 = sub_A745D0((_QWORD *)(a1 + 72), v7);
            if ( a4 >= v10 )
              v10 = a4;
            v20 = v10;
          }
          else
          {
            v20 = a4;
          }
          result = sub_A745B0((_QWORD *)(a1 + 72), v7);
          if ( v20 > result )
            break;
          if ( v17 == ++v4 )
            return result;
        }
        v11 = (__int64 *)sub_BD5C60(a1);
        *(_QWORD *)(a1 + 72) = sub_A7B980((__int64 *)(a1 + 72), v11, (int)v7 + 1, 90);
        if ( !sub_B2F070(v19, v9) || (unsigned __int8)sub_B49B80(a1, v7, 43) )
        {
          v12 = (__int64 *)sub_BD5C60(a1);
          *(_QWORD *)(a1 + 72) = sub_A7B980((__int64 *)(a1 + 72), v12, (int)v7 + 1, 91);
        }
        ++v4;
        v13 = (__int64 *)sub_BD5C60(a1);
        v14 = sub_A77A80(v13, v20);
        v21[0] = v7;
        v15 = v14;
        v16 = (__int64 *)sub_BD5C60(a1);
        result = sub_A7B660((__int64 *)(a1 + 72), v16, v21, 1, v15);
        *(_QWORD *)(a1 + 72) = result;
      }
      while ( v17 != v4 );
    }
  }
  return result;
}
