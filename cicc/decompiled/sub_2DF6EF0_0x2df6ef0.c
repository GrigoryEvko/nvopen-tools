// Function: sub_2DF6EF0
// Address: 0x2df6ef0
//
void __fastcall sub_2DF6EF0(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // rax
  unsigned __int8 v4; // dl
  _BYTE **v5; // rax
  unsigned __int8 v6; // dl
  __int64 *v7; // rax
  void *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  size_t v11; // r14
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // rax
  _DWORD *v15; // rdx
  _WORD *v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // r14
  unsigned int v19; // eax
  __int64 v20[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( !*a1 )
    return;
  v3 = (_BYTE *)sub_B10D00((__int64)a1);
  if ( *v3 == 16
    || ((v4 = *(v3 - 16), (v4 & 2) == 0)
      ? (v5 = (_BYTE **)&v3[-8 * ((v4 >> 2) & 0xF) - 16])
      : (v5 = (_BYTE **)*((_QWORD *)v3 - 4)),
        (v3 = *v5) != 0) )
  {
    v6 = *(v3 - 16);
    v7 = (v6 & 2) != 0 ? (__int64 *)*((_QWORD *)v3 - 4) : (__int64 *)&v3[-8 * ((v6 >> 2) & 0xF) - 16];
    if ( *v7 )
    {
      v8 = (void *)sub_B91420(*v7);
      v10 = *(_BYTE **)(a2 + 32);
      v11 = v9;
      if ( v9 <= *(_QWORD *)(a2 + 24) - (_QWORD)v10 )
      {
        if ( v9 )
        {
          memcpy(v10, v8, v9);
          v10 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
          *(_QWORD *)(a2 + 32) = v10;
        }
        goto LABEL_12;
      }
      sub_CB6200(a2, (unsigned __int8 *)v8, v9);
    }
  }
  v10 = *(_BYTE **)(a2 + 32);
LABEL_12:
  if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)v10 )
  {
    v12 = sub_CB5D20(a2, 58);
  }
  else
  {
    v12 = a2;
    *(_QWORD *)(a2 + 32) = v10 + 1;
    *v10 = 58;
  }
  v13 = sub_B10CE0((__int64)a1);
  sub_CB59D0(v12, v13);
  if ( (unsigned int)sub_B10CF0((__int64)a1) )
  {
    v17 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(a2 + 24) )
    {
      v18 = sub_CB5D20(a2, 58);
    }
    else
    {
      v18 = a2;
      *(_QWORD *)(a2 + 32) = v17 + 1;
      *v17 = 58;
    }
    v19 = sub_B10CF0((__int64)a1);
    sub_CB59D0(v18, v19);
  }
  v14 = sub_B10D40((__int64)a1);
  sub_B10CB0(v20, v14);
  if ( v20[0] )
  {
    v15 = *(_DWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v15 <= 3u )
    {
      sub_CB6200(a2, " @[ ", 4u);
    }
    else
    {
      *v15 = 542851104;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
    sub_2DF6EF0(v20, a2);
    v16 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 1u )
    {
      sub_CB6200(a2, (unsigned __int8 *)" ]", 2u);
    }
    else
    {
      *v16 = 23840;
      *(_QWORD *)(a2 + 32) += 2LL;
    }
    if ( v20[0] )
      sub_B91220((__int64)v20, v20[0]);
  }
}
