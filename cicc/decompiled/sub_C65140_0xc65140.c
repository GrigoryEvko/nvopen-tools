// Function: sub_C65140
// Address: 0xc65140
//
__int64 __fastcall sub_C65140(__int64 a1, int a2)
{
  int v3; // r14d
  _BYTE *v4; // rax
  char *v5; // rbx
  size_t v6; // r8
  char *v7; // r9
  int v8; // r12d
  _BYTE *v9; // rdi
  __int64 v10; // r10
  _BYTE *v11; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  size_t v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  char *v17; // [rsp+10h] [rbp-40h]
  size_t v18; // [rsp+10h] [rbp-40h]
  char v19; // [rsp+1Fh] [rbp-31h]

  v3 = a2;
  v4 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(a1 + 24) )
  {
    sub_CB5D20(a1, 40);
    if ( a2 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v4 + 1;
    *v4 = 40;
    if ( a2 )
    {
LABEL_3:
      v19 = 1;
      v5 = (char *)&unk_4979AB8;
      v6 = 3;
      v7 = "all";
      v8 = 1023;
      while ( 1 )
      {
        if ( (v3 & v8) != v8 )
        {
          if ( v5 == (char *)&unk_4979C20 )
            goto LABEL_13;
          goto LABEL_5;
        }
        v9 = *(_BYTE **)(a1 + 32);
        if ( v19 )
          break;
        if ( *(_BYTE **)(a1 + 24) != v9 )
        {
          *v9 = 32;
          v10 = a1;
          v9 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 1LL);
          *(_QWORD *)(a1 + 32) = v9;
LABEL_9:
          if ( *(_QWORD *)(v10 + 24) - (_QWORD)v9 < v6 )
            goto LABEL_18;
          goto LABEL_10;
        }
        v15 = v6;
        v17 = v7;
        v13 = sub_CB6200(a1, " ", 1);
        v6 = v15;
        v7 = v17;
        v10 = v13;
        v9 = *(_BYTE **)(v13 + 32);
        if ( *(_QWORD *)(v13 + 24) - (_QWORD)v9 < v15 )
        {
LABEL_18:
          sub_CB6200(v10, v7, v6);
          goto LABEL_12;
        }
LABEL_10:
        if ( v6 )
        {
          v16 = v10;
          v18 = v6;
          memcpy(v9, v7, v6);
          *(_QWORD *)(v16 + 32) += v18;
        }
LABEL_12:
        v3 = (unsigned __int16)v3 & (unsigned __int16)~(_WORD)v8 & 0x3FF;
        if ( v5 == (char *)&unk_4979C20 )
        {
LABEL_13:
          v11 = *(_BYTE **)(a1 + 32);
          if ( (unsigned __int64)v11 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, 41);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v11 + 1;
            *v11 = 41;
          }
          return a1;
        }
LABEL_5:
        v8 = *(_DWORD *)v5;
        v7 = (char *)*((_QWORD *)v5 + 1);
        v5 += 24;
        v6 = *((_QWORD *)v5 - 1);
      }
      v19 = 0;
      v10 = a1;
      goto LABEL_9;
    }
  }
  v14 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v14) <= 4 )
  {
    sub_CB6200(a1, "none)", 5);
  }
  else
  {
    *(_DWORD *)v14 = 1701736302;
    *(_BYTE *)(v14 + 4) = 41;
    *(_QWORD *)(a1 + 32) += 5LL;
  }
  return a1;
}
