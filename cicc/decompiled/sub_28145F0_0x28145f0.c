// Function: sub_28145F0
// Address: 0x28145f0
//
__int64 __fastcall sub_28145F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  char v10; // al
  __int64 result; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int8 *v19; // r8
  unsigned __int8 *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = sub_D4B130(a2);
  *(_QWORD *)(a1 + 8) = **(_QWORD **)(a2 + 32);
  *(_QWORD *)(a1 + 16) = sub_D46F00(a2);
  *(_QWORD *)(a1 + 24) = sub_D47470(a2);
  v9 = sub_D47930(a2);
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 32) = v9;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 56) = 0x1000000000LL;
  *(_QWORD *)(a1 + 200) = 0x1000000000LL;
  *(_BYTE *)(a1 + 336) = 1;
  *(_QWORD *)(a1 + 344) = sub_D4B890(a2);
  *(_QWORD *)(a1 + 352) = a6;
  v10 = sub_2A04CA0(a2);
  *(_QWORD *)(a1 + 368) = a3;
  *(_BYTE *)(a1 + 360) = v10;
  *(_BYTE *)(a1 + 361) = 0;
  *(_QWORD *)(a1 + 376) = a4;
  *(_QWORD *)(a1 + 384) = a5;
  result = *(_QWORD *)(a2 + 40);
  v23 = result;
  v25 = *(_QWORD *)(a2 + 32);
  if ( result != v25 )
  {
    while ( 1 )
    {
      result = v25;
      v12 = *(_QWORD *)v25;
      if ( (*(_WORD *)(*(_QWORD *)v25 + 2LL) & 0x7FFF) != 0 )
        break;
      v13 = *(_QWORD *)(v12 + 56);
      v14 = v12 + 48;
      if ( v13 != v14 )
      {
        while ( 1 )
        {
          v19 = (unsigned __int8 *)(v13 - 24);
          if ( !v13 )
            v19 = 0;
          v20 = v19;
          result = sub_B46790(v19, 0);
          if ( (_BYTE)result )
            goto LABEL_15;
          result = *v20;
          if ( (_BYTE)result == 62 )
          {
            if ( (v20[2] & 1) != 0 )
              goto LABEL_15;
          }
          else if ( (_BYTE)result == 61 && (v20[2] & 1) != 0 )
          {
            goto LABEL_15;
          }
          if ( (unsigned __int8)sub_B46490((__int64)v20) )
          {
            v22 = *(unsigned int *)(a1 + 200);
            if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 204) )
            {
              sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v22 + 1, 8u, v15, v16);
              v22 = *(unsigned int *)(a1 + 200);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v22) = v20;
            ++*(_DWORD *)(a1 + 200);
          }
          if ( (unsigned __int8)sub_B46420((__int64)v20) )
          {
            v21 = *(unsigned int *)(a1 + 56);
            if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
            {
              sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v21 + 1, 8u, v17, v18);
              v21 = *(unsigned int *)(a1 + 56);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v21) = v20;
            ++*(_DWORD *)(a1 + 56);
            v13 = *(_QWORD *)(v13 + 8);
            if ( v14 == v13 )
              break;
          }
          else
          {
            v13 = *(_QWORD *)(v13 + 8);
            if ( v14 == v13 )
              break;
          }
        }
      }
      v25 += 8;
      result = v25;
      if ( v23 == v25 )
        return result;
    }
LABEL_15:
    *(_DWORD *)(a1 + 200) = 0;
    *(_DWORD *)(a1 + 56) = 0;
    *(_BYTE *)(a1 + 336) = 0;
  }
  return result;
}
