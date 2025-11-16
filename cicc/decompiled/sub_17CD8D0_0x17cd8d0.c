// Function: sub_17CD8D0
// Address: 0x17cd8d0
//
__int64 *__fastcall sub_17CD8D0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  __int64 v7; // rdx
  __int64 v8; // rdi
  char v9; // al
  unsigned int v10; // eax
  __int64 v11; // rbx
  __int64 *v12; // rax
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r15
  int v18; // r8d
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdx
  _BYTE *v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // [rsp+0h] [rbp-70h]
  _BYTE *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _BYTE v27[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v4 > 0xFu || (v7 = 35454, !_bittest64(&v7, v4)) )
  {
    if ( (unsigned int)(v4 - 13) > 1 && (_DWORD)v4 != 16 || !sub_16435F0(a2, 0) )
      return 0;
    LOBYTE(v4) = *(_BYTE *)(a2 + 8);
  }
  v5 = a2;
  if ( (_BYTE)v4 == 11 )
    return (__int64 *)v5;
  v8 = sub_1632FA0(*(_QWORD *)(*a1 + 40LL));
  v9 = *(_BYTE *)(a2 + 8);
  switch ( v9 )
  {
    case 16:
      v10 = sub_127FA20(v8, *(_QWORD *)(a2 + 24));
      v11 = *(_QWORD *)(a2 + 32);
      v12 = (__int64 *)sub_1644900(*(_QWORD **)(a1[1] + 168LL), v10);
      return sub_16463B0(v12, v11);
    case 14:
      v13 = *(_QWORD *)(a2 + 32);
      v14 = (__int64 *)sub_17CD8D0(a1, *(_QWORD *)(a2 + 24));
      return sub_1645D80(v14, v13);
    case 13:
      v15 = *(unsigned int *)(a2 + 12);
      v25 = v27;
      v26 = 0x400000000LL;
      if ( (_DWORD)v15 )
      {
        v16 = 8 * v15;
        v17 = 0;
        do
        {
          v19 = sub_17CD8D0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v17));
          v20 = (unsigned int)v26;
          if ( (unsigned int)v26 >= HIDWORD(v26) )
          {
            v24 = v19;
            sub_16CD150((__int64)&v25, v27, 0, 8, v18, v19);
            v20 = (unsigned int)v26;
            v19 = v24;
          }
          v17 += 8;
          *(_QWORD *)&v25[8 * v20] = v19;
          v21 = (unsigned int)(v26 + 1);
          LODWORD(v26) = v26 + 1;
        }
        while ( v16 != v17 );
        v22 = v25;
      }
      else
      {
        v21 = 0;
        v22 = v27;
      }
      v5 = sub_1645600(*(_QWORD **)(a1[1] + 168LL), v22, v21, (*(_DWORD *)(a2 + 8) & 0x200) != 0);
      if ( v25 != v27 )
        _libc_free((unsigned __int64)v25);
      return (__int64 *)v5;
  }
  v23 = sub_127FA20(v8, a2);
  return (__int64 *)sub_1644900(*(_QWORD **)(a1[1] + 168LL), v23);
}
