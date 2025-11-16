// Function: sub_2463540
// Address: 0x2463540
//
_QWORD *__fastcall sub_2463540(__int64 *a1, __int64 a2)
{
  int v3; // edx
  __int64 v5; // rdi
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // esi
  int v10; // eax
  __int64 *v11; // rax
  _QWORD *result; // rax
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  unsigned __int64 v21; // r9
  __int64 v22; // rdx
  _BYTE *v23; // rsi
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-70h]
  _QWORD *v27; // [rsp+8h] [rbp-68h]
  _BYTE *v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+18h] [rbp-58h]
  _BYTE v30[80]; // [rsp+20h] [rbp-50h] BYREF

  v3 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v3 == 12 )
    return (_QWORD *)a2;
  if ( (unsigned __int8)v3 > 3u && (_BYTE)v3 != 5 && (v3 & 0xFB) != 0xA && (v3 & 0xFD) != 4 )
  {
    if ( (unsigned __int8)(v3 - 15) > 3u && v3 != 20 || !(unsigned __int8)sub_BCEBA0(a2, 0) )
      return 0;
    if ( *(_BYTE *)(a2 + 8) == 12 )
      return (_QWORD *)a2;
  }
  v5 = sub_B2BEC0(*a1);
  v6 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v6 - 17) > 1 )
  {
    if ( (_BYTE)v6 == 16 )
    {
      v13 = *(_QWORD *)(a2 + 32);
      v14 = (__int64 *)sub_2463540(a1, *(_QWORD *)(a2 + 24));
      return sub_BCD420(v14, v13);
    }
    else if ( (_BYTE)v6 == 15 )
    {
      v15 = *(unsigned int *)(a2 + 12);
      v28 = v30;
      v29 = 0x400000000LL;
      if ( (_DWORD)v15 )
      {
        v16 = 8 * v15;
        v17 = 0;
        do
        {
          v18 = sub_2463540(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v17));
          v20 = (unsigned int)v29;
          v21 = (unsigned int)v29 + 1LL;
          if ( v21 > HIDWORD(v29) )
          {
            v26 = v18;
            sub_C8D5F0((__int64)&v28, v30, (unsigned int)v29 + 1LL, 8u, v19, v21);
            v20 = (unsigned int)v29;
            v18 = v26;
          }
          v17 += 8;
          *(_QWORD *)&v28[8 * v20] = v18;
          v22 = (unsigned int)(v29 + 1);
          LODWORD(v29) = v29 + 1;
        }
        while ( v16 != v17 );
        v23 = v28;
      }
      else
      {
        v22 = 0;
        v23 = v30;
      }
      result = sub_BD0B90(*(_QWORD **)(a1[1] + 72), v23, v22, (*(_DWORD *)(a2 + 8) & 0x200) != 0);
      if ( v28 != v30 )
      {
        v27 = result;
        _libc_free((unsigned __int64)v28);
        return v27;
      }
    }
    else
    {
      v28 = (_BYTE *)sub_9208B0(v5, a2);
      v29 = v24;
      v25 = sub_CA1930(&v28);
      return (_QWORD *)sub_BCCE00(*(_QWORD **)(a1[1] + 72), v25);
    }
  }
  else
  {
    v7 = sub_9208B0(v5, *(_QWORD *)(a2 + 24));
    v29 = v8;
    v28 = (_BYTE *)v7;
    v9 = sub_CA1930(&v28);
    v10 = *(_DWORD *)(a2 + 32);
    BYTE4(v28) = *(_BYTE *)(a2 + 8) == 18;
    LODWORD(v28) = v10;
    v11 = (__int64 *)sub_BCCE00(*(_QWORD **)(a1[1] + 72), v9);
    return (_QWORD *)sub_BCE1B0(v11, (__int64)v28);
  }
  return result;
}
