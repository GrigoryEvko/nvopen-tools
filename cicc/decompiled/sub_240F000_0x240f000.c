// Function: sub_240F000
// Address: 0x240f000
//
_QWORD *__fastcall sub_240F000(__int64 a1, __int64 a2)
{
  int v2; // edx
  unsigned __int8 v3; // al
  _QWORD *result; // rax
  __int64 v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rdx
  unsigned __int64 v13; // r9
  __int64 v14; // rdx
  _BYTE *v15; // rsi
  __int64 v16; // [rsp+0h] [rbp-70h]
  _QWORD *v17; // [rsp+8h] [rbp-68h]
  _BYTE *v18; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h]
  _BYTE v20[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v2 == 12 )
    return *(_QWORD **)(a1 + 48);
  v3 = *(_BYTE *)(a2 + 8);
  if ( (unsigned __int8)v2 > 3u )
  {
    if ( (_BYTE)v2 != 5 )
    {
      if ( (v2 & 0xFD) == 4 || (v2 & 0xFB) == 0xA )
        goto LABEL_3;
      if ( (unsigned __int8)(v2 - 15) <= 3u || v2 == 20 )
      {
        if ( (unsigned __int8)sub_BCEBA0(a2, 0) )
        {
          v3 = *(_BYTE *)(a2 + 8);
          if ( v3 != 12 )
          {
            v2 = v3;
            goto LABEL_3;
          }
        }
      }
    }
    return *(_QWORD **)(a1 + 48);
  }
LABEL_3:
  if ( (unsigned int)(v2 - 17) <= 1 )
    return *(_QWORD **)(a1 + 48);
  if ( v3 == 16 )
  {
    v5 = *(_QWORD *)(a2 + 32);
    v6 = (__int64 *)sub_240F000(a1, *(_QWORD *)(a2 + 24));
    return sub_BCD420(v6, v5);
  }
  if ( v3 != 15 )
    return *(_QWORD **)(a1 + 48);
  v7 = *(unsigned int *)(a2 + 12);
  v18 = v20;
  v19 = 0x400000000LL;
  if ( (_DWORD)v7 )
  {
    v8 = 8 * v7;
    v9 = 0;
    do
    {
      v10 = sub_240F000(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v9));
      v12 = (unsigned int)v19;
      v13 = (unsigned int)v19 + 1LL;
      if ( v13 > HIDWORD(v19) )
      {
        v16 = v10;
        sub_C8D5F0((__int64)&v18, v20, (unsigned int)v19 + 1LL, 8u, v11, v13);
        v12 = (unsigned int)v19;
        v10 = v16;
      }
      v9 += 8;
      *(_QWORD *)&v18[8 * v12] = v10;
      v14 = (unsigned int)(v19 + 1);
      LODWORD(v19) = v19 + 1;
    }
    while ( v9 != v8 );
    v15 = v18;
  }
  else
  {
    v14 = 0;
    v15 = v20;
  }
  result = sub_BD0B90(*(_QWORD **)(a1 + 8), v15, v14, 0);
  if ( v18 != v20 )
  {
    v17 = result;
    _libc_free((unsigned __int64)v18);
    return v17;
  }
  return result;
}
