// Function: sub_305F550
// Address: 0x305f550
//
__int64 __fastcall sub_305F550(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rcx
  int v4; // r14d
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  signed __int64 v17; // [rsp+8h] [rbp-58h]
  __int16 v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)a2;
  v17 = 1;
  v3 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), (__int64 *)a2, 0);
  v4 = v3;
  v6 = v5;
  while ( 1 )
  {
    LOWORD(v3) = v4;
    sub_2FE6CC0((__int64)&v18, *(_QWORD *)(a1 + 32), v2, v3, v6);
    if ( (_BYTE)v18 == 10 )
      return 0;
    if ( !(_BYTE)v18 )
    {
LABEL_7:
      v10 = (unsigned int)(v4 - 17);
      goto LABEL_8;
    }
    if ( (v18 & 0xFB) == 2 )
    {
      v14 = 2 * v17;
      if ( !is_mul_ok(2u, v17) )
      {
        v14 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v17 <= 0 )
          v14 = 0x8000000000000000LL;
      }
      v17 = v14;
    }
    if ( (_WORD)v19 == (_WORD)v4 )
    {
      if ( (_WORD)v4 )
        goto LABEL_7;
      if ( v6 == v20 )
        break;
    }
    v3 = v19;
    v6 = v20;
    v4 = (unsigned __int16)v19;
  }
  v10 = 4294967279LL;
LABEL_8:
  v11 = 0;
  if ( *(_BYTE *)(a2 + 8) == 17 )
    v11 = a2;
  v12 = v11;
  if ( (unsigned __int16)v10 <= 0x9Eu
    && ((v15 = *(_DWORD *)(v11 + 32)) == 0 || (v10 = (unsigned int)(v15 - 1), (v15 & (unsigned int)v10) != 0))
    && (v18 = v4, v19 = 0, (v16 = sub_3007410((__int64)&v18, *(__int64 **)a2, v10, v7, v8, v9)) != 0)
    && *(_BYTE *)(v16 + 8) == 17
    && *(_QWORD *)(v16 + 24) == *(_QWORD *)(v12 + 24) )
  {
    return (*(_DWORD *)(v12 + 32) != 0)
         + (*(_DWORD *)(v12 + 32) - (unsigned int)(*(_DWORD *)(v12 + 32) != 0)) / *(_DWORD *)(v16 + 32);
  }
  else
  {
    return (unsigned int)v17;
  }
}
