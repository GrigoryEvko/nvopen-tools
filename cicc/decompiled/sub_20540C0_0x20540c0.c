// Function: sub_20540C0
// Address: 0x20540c0
//
unsigned __int64 __fastcall sub_20540C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 v6; // rcx
  int v7; // r8d
  unsigned int v8; // edx
  __int64 v9; // rsi
  __int64 *v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r14
  _QWORD *v15; // rbx
  __int64 v16; // rax
  int v17; // eax
  _QWORD *v18; // rax
  int v19; // r8d
  __int64 *v20; // r9
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rbx
  __int64 v24; // rsi
  __int64 *v25; // [rsp+8h] [rbp-78h]
  __int64 *v26; // [rsp+18h] [rbp-68h]
  unsigned int v28; // [rsp+28h] [rbp-58h]
  unsigned int v29; // [rsp+2Ch] [rbp-54h]
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32[7]; // [rsp+48h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 96);
  v30 = a3;
  v31 = a4;
  if ( (_DWORD)result )
  {
    v6 = *(_QWORD *)(a1 + 80);
    v7 = 1;
    v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v25 = (__int64 *)(v6 + 32LL * v8);
    v9 = *v25;
    if ( a2 == *v25 )
    {
LABEL_3:
      result = v6 + 32 * result;
      if ( v25 != (__int64 *)result )
      {
        result = v25[2];
        v26 = (__int64 *)result;
        if ( result != v25[1] )
        {
          v10 = (__int64 *)v25[1];
          do
          {
            v11 = v10[1];
            v12 = *v10;
            v32[0] = v11;
            if ( v11 )
              sub_1623A60((__int64)v32, v11, 2);
            v13 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            v29 = *((_DWORD *)v10 + 4);
            v28 = *(_DWORD *)(v30 + 64);
            v14 = *(_QWORD *)(*(_QWORD *)(v12 + 24 * (1 - v13)) + 24LL);
            v15 = *(_QWORD **)(*(_QWORD *)(v12 + 24 * (2 - v13)) + 24LL);
            v16 = sub_15C70A0((__int64)v32);
            if ( !(unsigned __int8)sub_2054040(a1, a2, v14, v15, v16, 0, &v30) )
            {
              v17 = v28;
              if ( v29 >= v28 )
                v17 = v29;
              v18 = sub_2054060(a1, v30, v31, v14, (__int64)v15, v32, v17);
              sub_1D30360(*(_QWORD *)(a1 + 552), (__int64)v18, v30, 0, v19, v20);
            }
            if ( v32[0] )
              sub_161E7C0((__int64)v32, v32[0]);
            v10 += 3;
          }
          while ( v26 != v10 );
          result = (unsigned __int64)v25;
          v21 = v25[1];
          v22 = v25[2];
          if ( v21 != v22 )
          {
            v23 = v25[1];
            do
            {
              v24 = *(_QWORD *)(v23 + 8);
              if ( v24 )
                sub_161E7C0(v23 + 8, v24);
              v23 += 24;
            }
            while ( v23 != v22 );
            result = (unsigned __int64)v25;
            v25[2] = v21;
          }
        }
      }
    }
    else
    {
      while ( v9 != -8 )
      {
        v8 = (result - 1) & (v7 + v8);
        v25 = (__int64 *)(v6 + 32LL * v8);
        v9 = *v25;
        if ( a2 == *v25 )
          goto LABEL_3;
        ++v7;
      }
    }
  }
  return result;
}
