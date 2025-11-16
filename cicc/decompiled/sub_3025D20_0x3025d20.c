// Function: sub_3025D20
// Address: 0x3025d20
//
__int64 __fastcall sub_3025D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  char v10; // al
  __int64 v11; // rdx
  bool v12; // zf
  __int64 v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _WORD *v16; // rdx
  __int64 v18; // rdx
  unsigned __int8 *v19; // [rsp+0h] [rbp-50h] BYREF
  size_t v20; // [rsp+8h] [rbp-48h]
  _QWORD v21[2]; // [rsp+10h] [rbp-40h] BYREF
  char v22; // [rsp+20h] [rbp-30h]

  if ( sub_B2FC80(a2) )
  {
    sub_3022230((__int64)&v19, a2);
    if ( v22 )
    {
      sub_CB6200(a4, v19, v20);
      if ( v22 )
      {
        v22 = 0;
        if ( v19 != (unsigned __int8 *)v21 )
          j_j___libc_free_0((unsigned __int64)v19);
      }
    }
  }
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 1280LL) == 1 )
    sub_3022060((_BYTE *)a2, a4, v6, v7, v8, v9);
  v10 = sub_CE9220(a2);
  v11 = *(_QWORD *)(a4 + 32);
  v12 = v10 == 0;
  v13 = *(_QWORD *)(a4 + 24);
  if ( v12 )
  {
    if ( (unsigned __int64)(v13 - v11) <= 5 )
    {
      sub_CB6200(a4, ".func ", 6u);
    }
    else
    {
      *(_DWORD *)v11 = 1853187630;
      *(_WORD *)(v11 + 4) = 8291;
      *(_QWORD *)(a4 + 32) += 6LL;
    }
  }
  else
  {
    if ( (unsigned __int64)(v13 - v11) > 6 )
    {
      *(_DWORD *)v11 = 1953391918;
      *(_WORD *)(v11 + 4) = 31090;
      *(_BYTE *)(v11 + 6) = 32;
      *(_QWORD *)(a4 + 32) += 7LL;
      if ( !(unsigned __int8)sub_CE9620(a2) )
        goto LABEL_11;
      goto LABEL_7;
    }
    sub_CB6200(a4, ".entry ", 7u);
  }
  if ( (unsigned __int8)sub_CE9620(a2) )
LABEL_7:
    sub_3021B30(a2, a4);
LABEL_11:
  sub_3022420(a1, **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL), a2, a4);
  sub_EA12C0(a3, a4, *(_BYTE **)(a1 + 208));
  v14 = *(_BYTE **)(a4 + 32);
  if ( *(_BYTE **)(a4 + 24) == v14 )
  {
    sub_CB6200(a4, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v14 = 10;
    ++*(_QWORD *)(a4 + 32);
  }
  sub_3024DF0(a1, *(_QWORD *)(a2 + 24), a2, a4);
  v15 = *(_BYTE **)(a4 + 32);
  if ( *(_BYTE **)(a4 + 24) == v15 )
  {
    sub_CB6200(a4, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v15 = 10;
    ++*(_QWORD *)(a4 + 32);
  }
  if ( (unsigned __int8)sub_307AAA0(a2, *(_QWORD *)(a1 + 200)) )
  {
    v18 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v18) <= 8 )
    {
      sub_CB6200(a4, (unsigned __int8 *)".noreturn", 9u);
    }
    else
    {
      *(_BYTE *)(v18 + 8) = 110;
      *(_QWORD *)v18 = 0x72757465726F6E2ELL;
      *(_QWORD *)(a4 + 32) += 9LL;
    }
  }
  if ( sub_B2FC80(a2) )
  {
    sub_314D260(&v19, a2, 0);
    if ( v22 )
    {
      sub_CB6200(a4, v19, v20);
      if ( v22 )
      {
        v22 = 0;
        if ( v19 != (unsigned __int8 *)v21 )
          j_j___libc_free_0((unsigned __int64)v19);
      }
    }
  }
  v16 = *(_WORD **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v16 <= 1u )
    return sub_CB6200(a4, (unsigned __int8 *)";\n", 2u);
  *v16 = 2619;
  *(_QWORD *)(a4 + 32) += 2LL;
  return 2619;
}
