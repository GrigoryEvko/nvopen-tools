// Function: sub_301F8B0
// Address: 0x301f8b0
//
char __fastcall sub_301F8B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  _QWORD *v9; // r15
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  void *v15; // rdx
  void *v16; // rdi
  unsigned __int8 *v17; // rsi
  size_t v18; // r13

  v9 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 168LL);
  if ( sub_301F2A0(v9, a2) )
  {
    v12 = *(_QWORD *)(v10 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v12) <= 2 )
    {
      sub_CB6200(v10, "\t}\n", 3u);
    }
    else
    {
      *(_BYTE *)(v12 + 2) = 10;
      *(_WORD *)v12 = 32009;
      *(_QWORD *)(v10 + 32) += 3LL;
    }
  }
  LOBYTE(v11) = sub_301F2A0(v9, a3);
  if ( (_BYTE)v11 )
  {
    sub_301F800(a1);
    v13 = *(_QWORD *)(a5 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a5 + 24) - v13) <= 8 )
    {
      sub_CB6200(a5, "\t.section", 9u);
    }
    else
    {
      *(_BYTE *)(v13 + 8) = 110;
      *(_QWORD *)v13 = 0x6F69746365732E09LL;
      *(_QWORD *)(a5 + 32) += 9LL;
    }
    (**(void (__fastcall ***)(__int64, _QWORD, __int64, __int64, _QWORD))a3)(
      a3,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 152LL),
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 24LL,
      a5,
      a4);
    v14 = *(_QWORD *)(a5 + 32);
    v11 = *(_QWORD *)(a5 + 24) - v14;
    if ( v11 <= 2 )
    {
      LOBYTE(v11) = sub_CB6200(a5, "\t{\n", 3u);
    }
    else
    {
      *(_BYTE *)(v14 + 2) = 10;
      *(_WORD *)v14 = 31497;
      *(_QWORD *)(a5 + 32) += 3LL;
    }
    *(_BYTE *)(a1 + 160) = 1;
  }
  else
  {
    if ( v9[113] == a3 )
    {
      v15 = *(void **)(a5 + 32);
      if ( *(_QWORD *)(a5 + 24) - (_QWORD)v15 <= 9u )
      {
        sub_CB6200(a5, "\t.section ", 0xAu);
        v16 = *(void **)(a5 + 32);
      }
      else
      {
        qmemcpy(v15, "\t.section ", 10);
        v16 = (void *)(*(_QWORD *)(a5 + 32) + 10LL);
        *(_QWORD *)(a5 + 32) = v16;
      }
      v17 = *(unsigned __int8 **)(a3 + 128);
      v18 = *(_QWORD *)(a3 + 136);
      v11 = *(_QWORD *)(a5 + 24) - (_QWORD)v16;
      if ( v18 > v11 )
      {
        LOBYTE(v11) = sub_CB6200(a5, v17, v18);
      }
      else if ( v18 )
      {
        LOBYTE(v11) = (unsigned __int8)memcpy(v16, v17, v18);
        *(_QWORD *)(a5 + 32) += v18;
      }
    }
    *(_BYTE *)(a1 + 160) = 0;
  }
  return v11;
}
