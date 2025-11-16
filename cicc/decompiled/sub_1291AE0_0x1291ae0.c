// Function: sub_1291AE0
// Address: 0x1291ae0
//
_QWORD *__fastcall sub_1291AE0(__int64 a1, unsigned __int64 a2)
{
  __int64 *v4; // r14
  __int64 v5; // rdx
  unsigned int v6; // ebx
  char *v7; // r15
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rdi
  unsigned __int64 *v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 *v17; // r14
  __int64 *i; // rbx
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  bool v22; // al
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  v4 = *(__int64 **)(a2 + 48);
  if ( *(_QWORD *)(a1 + 136) )
  {
    if ( v4 )
    {
      if ( sub_127B420(*v4) )
      {
        sub_12A6C40(a1, v4, *(_QWORD *)(a1 + 136), *(unsigned int *)(a1 + 144), 0);
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 136);
        v6 = unk_4D0463C;
        if ( unk_4D0463C )
        {
          v22 = sub_126A420(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 136));
          v5 = *(_QWORD *)(a1 + 136);
          v6 = v22;
        }
        v23 = v5;
        v7 = sub_128F980(a1, (__int64)v4);
        LOWORD(v27) = 257;
        v8 = sub_1648A60(64, 2);
        v9 = (_QWORD *)v8;
        if ( v8 )
          sub_15F9650(v8, v7, v23, v6, 0);
        v10 = *(_QWORD *)(a1 + 56);
        if ( v10 )
        {
          v11 = *(unsigned __int64 **)(a1 + 64);
          sub_157E9D0(v10 + 40, v9);
          v12 = v9[3];
          v13 = *v11;
          v9[4] = v11;
          v13 &= 0xFFFFFFFFFFFFFFF8LL;
          v9[3] = v13 | v12 & 7;
          *(_QWORD *)(v13 + 8) = v9 + 3;
          *v11 = *v11 & 7 | (unsigned __int64)(v9 + 3);
        }
        sub_164B780(v9, &v25);
        v14 = *(_QWORD *)(a1 + 48);
        if ( v14 )
        {
          v24 = *(_QWORD *)(a1 + 48);
          sub_1623A60(&v24, v14, 2);
          if ( v9[6] )
            sub_161E7C0(v9 + 6);
          v15 = v24;
          v9[6] = v24;
          if ( v15 )
            sub_1623210(&v24, v15, v9 + 6);
        }
        sub_15F9450(v9, *(unsigned int *)(a1 + 144));
      }
    }
  }
  else if ( v4 )
  {
    sub_127FF60((__int64)&v25, a1, *(__int64 **)(a2 + 48), 0, 0, 0);
  }
  if ( *(_BYTE *)(a1 + 168) )
  {
    sub_1291800(&v25, a1, a2);
    v17 = v26;
    for ( i = v25; v17 != i; ++i )
    {
      v19 = *i;
      sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(v19 + 120));
      sub_12A5710(a1, v19, 1);
    }
    v20 = *(_QWORD *)(a1 + 352);
    v21 = *(_QWORD *)(v20 - 24);
    if ( v21 != *(_QWORD *)(v20 - 16) )
      *(_QWORD *)(v20 - 16) = v21;
    if ( v25 )
      j_j___libc_free_0(v25, v27 - (_QWORD)v25);
  }
  return sub_12909B0((_QWORD *)a1, *(_QWORD *)(a1 + 128));
}
