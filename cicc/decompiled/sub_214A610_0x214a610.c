// Function: sub_214A610
// Address: 0x214a610
//
char __fastcall sub_214A610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // r14
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  void *v15; // rdx
  void *v16; // rdi
  char *v17; // rsi
  size_t v18; // r13

  v9 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 32LL);
  if ( sub_2149FD0(v9, a2) )
  {
    v12 = *(_QWORD *)(v10 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v12) <= 2 )
    {
      sub_16E7EE0(v10, "\t}\n", 3u);
    }
    else
    {
      *(_BYTE *)(v12 + 2) = 10;
      *(_WORD *)v12 = 32009;
      *(_QWORD *)(v10 + 24) += 3LL;
    }
  }
  LOBYTE(v11) = sub_2149FD0(v9, a3);
  if ( (_BYTE)v11 )
  {
    sub_214A560(a1);
    v13 = *(_QWORD *)(a5 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a5 + 16) - v13) <= 8 )
    {
      sub_16E7EE0(a5, "\t.section", 9u);
    }
    else
    {
      *(_BYTE *)(v13 + 8) = 110;
      *(_QWORD *)v13 = 0x6F69746365732E09LL;
      *(_QWORD *)(a5 + 24) += 9LL;
    }
    (**(void (__fastcall ***)(__int64, _QWORD, __int64, __int64, __int64))a3)(
      a3,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 16LL),
      (__int64)(v9 + 87),
      a5,
      a4);
    v14 = *(_QWORD *)(a5 + 24);
    v11 = *(_QWORD *)(a5 + 16) - v14;
    if ( v11 <= 2 )
    {
      LOBYTE(v11) = sub_16E7EE0(a5, "\t{\n", 3u);
    }
    else
    {
      *(_BYTE *)(v14 + 2) = 10;
      *(_WORD *)v14 = 31497;
      *(_QWORD *)(a5 + 24) += 3LL;
    }
    *(_BYTE *)(a1 + 160) = 1;
  }
  else
  {
    if ( v9[84] == a3 )
    {
      v15 = *(void **)(a5 + 24);
      if ( *(_QWORD *)(a5 + 16) - (_QWORD)v15 <= 9u )
      {
        sub_16E7EE0(a5, "\t.section ", 0xAu);
        v16 = *(void **)(a5 + 24);
      }
      else
      {
        qmemcpy(v15, "\t.section ", 10);
        v16 = (void *)(*(_QWORD *)(a5 + 24) + 10LL);
        *(_QWORD *)(a5 + 24) = v16;
      }
      v17 = *(char **)(a3 + 152);
      v18 = *(_QWORD *)(a3 + 160);
      v11 = *(_QWORD *)(a5 + 16) - (_QWORD)v16;
      if ( v18 > v11 )
      {
        LOBYTE(v11) = sub_16E7EE0(a5, v17, v18);
      }
      else if ( v18 )
      {
        LOBYTE(v11) = (unsigned __int8)memcpy(v16, v17, v18);
        *(_QWORD *)(a5 + 24) += v18;
      }
    }
    *(_BYTE *)(a1 + 160) = 0;
  }
  return v11;
}
