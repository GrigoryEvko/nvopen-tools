// Function: sub_3184A90
// Address: 0x3184a90
//
__int64 __fastcall sub_3184A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int8 **v8; // rdx
  __int64 v9; // rax
  unsigned __int8 **v10; // rbx
  __int64 v11; // rcx
  unsigned __int8 **v12; // r13
  char v13; // al
  unsigned __int8 *v14; // r15
  unsigned __int8 **v15; // rax
  unsigned int v16; // r15d
  __int64 v18; // r14
  __int64 v19; // r9
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int8 **v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+20h] [rbp-60h]
  int v27; // [rsp+28h] [rbp-58h]
  char v28; // [rsp+2Ch] [rbp-54h]
  _BYTE v29[80]; // [rsp+30h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( *(_BYTE *)a3 == 84 && *(_QWORD *)(a3 + 40) == *(_QWORD *)(a2 + 40) )
  {
    v18 = 0;
    v23 = 8LL * (unsigned int)v7;
    if ( !(_DWORD)v7 )
      return 0;
    while ( 1 )
    {
      v19 = *(_QWORD *)(a2 - 8);
      v20 = *(_QWORD *)(a3 - 8);
      v21 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 0 )
      {
        v22 = 0;
        do
        {
          if ( *(_QWORD *)(v19 + 32LL * *(unsigned int *)(a2 + 72) + v18) == *(_QWORD *)(v20
                                                                                       + 32LL
                                                                                       * *(unsigned int *)(a3 + 72)
                                                                                       + 8 * v22) )
          {
            v21 = 32 * v22;
            goto LABEL_30;
          }
          ++v22;
        }
        while ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != (_DWORD)v22 );
        v21 = 0x1FFFFFFFE0LL;
      }
LABEL_30:
      v16 = sub_31843D0(a1, *(unsigned __int8 **)(v19 + 4 * v18), *(unsigned __int8 **)(v20 + v21));
      if ( (_BYTE)v16 )
        return v16;
      v18 += 8;
      if ( v18 == v23 )
        return 0;
    }
  }
  v8 = (unsigned __int8 **)v29;
  v28 = 1;
  v9 = 32 * v7;
  v24 = 0;
  v25 = (unsigned __int8 **)v29;
  v26 = 4;
  v27 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v11 = *(_QWORD *)(a2 - 8);
    v10 = (unsigned __int8 **)(v11 + v9);
  }
  else
  {
    v10 = (unsigned __int8 **)a2;
    v11 = a2 - v9;
  }
  if ( (unsigned __int8 **)v11 == v10 )
    return 0;
  v12 = (unsigned __int8 **)v11;
  v13 = 1;
  while ( 1 )
  {
    v14 = *v12;
    if ( !v13 )
      goto LABEL_6;
    v8 = &v25[HIDWORD(v26)];
    v15 = v25;
    if ( v25 != v8 )
      break;
LABEL_15:
    if ( HIDWORD(v26) < (unsigned int)v26 )
    {
      ++HIDWORD(v26);
      *v8 = v14;
      ++v24;
LABEL_17:
      v16 = sub_31843D0(a1, v14, (unsigned __int8 *)a3);
      v13 = v28;
      if ( (_BYTE)v16 )
        goto LABEL_18;
      goto LABEL_7;
    }
LABEL_6:
    sub_C8CC70((__int64)&v24, (__int64)v14, (__int64)v8, v11, a5, a6);
    v13 = v28;
    if ( (_BYTE)v8 )
      goto LABEL_17;
LABEL_7:
    v12 += 4;
    if ( v10 == v12 )
    {
      v16 = 0;
LABEL_18:
      if ( !v13 )
        _libc_free((unsigned __int64)v25);
      return v16;
    }
  }
  while ( 1 )
  {
    while ( *v15 != v14 )
    {
      if ( v8 == ++v15 )
        goto LABEL_15;
    }
    v12 += 4;
    if ( v10 == v12 )
      return 0;
    v14 = *v12;
    v15 = v25;
    if ( v25 == v8 )
      goto LABEL_15;
  }
}
