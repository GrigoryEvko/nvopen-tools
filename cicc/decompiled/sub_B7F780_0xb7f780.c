// Function: sub_B7F780
// Address: 0xb7f780
//
__int64 __fastcall sub_B7F780(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  const void *v3; // r13
  size_t v4; // rdx
  size_t v5; // r14
  int v6; // edx
  _DWORD *v7; // rax
  __int64 v8; // rax
  char *v9; // rsi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  _BYTE *v14; // rdi
  _BYTE *v15; // rax
  __int64 v17; // rax
  void *v18; // rdx
  __int64 v19; // r12
  _BYTE *v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rax
  void *v23; // [rsp+0h] [rbp-50h] BYREF
  const char *v24; // [rsp+8h] [rbp-48h]
  int v25; // [rsp+10h] [rbp-40h]

  if ( (_DWORD)qword_4F81AA8 == -1 && qword_4F819A8 == qword_4F819B0 )
    return 1;
  v3 = (const void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = v4;
  if ( !(_BYTE)a2 )
  {
    v6 = dword_4F81904 + 1;
    dword_4F81904 = v6;
    LOBYTE(v2) = v6 <= (int)qword_4F81AA8 || (_DWORD)qword_4F81AA8 == -1;
    if ( (_BYTE)v2 )
    {
      v7 = (_DWORD *)qword_4F819A8;
      if ( qword_4F819B0 == qword_4F819A8 )
      {
LABEL_5:
        v8 = sub_CB72A0(a1, a2);
        v24 = "%2d: ";
        v23 = &unk_49DAA58;
        v25 = dword_4F81904;
        v9 = "ENABLED   ";
        v10 = sub_CB6620(v8, &v23);
LABEL_9:
        v12 = *(_QWORD *)(v10 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v12) > 9 )
        {
          *(_QWORD *)v12 = *(_QWORD *)v9;
          *(_WORD *)(v12 + 8) = *((_WORD *)v9 + 4);
          v14 = (_BYTE *)(*(_QWORD *)(v10 + 32) + 10LL);
          *(_QWORD *)(v10 + 32) = v14;
        }
        else
        {
          v13 = sub_CB6200(v10, v9, 10);
          v14 = *(_BYTE **)(v13 + 32);
          v10 = v13;
        }
        v15 = *(_BYTE **)(v10 + 24);
        if ( v5 > v15 - v14 )
        {
          v10 = sub_CB6200(v10, v3, v5);
          v15 = *(_BYTE **)(v10 + 24);
          v14 = *(_BYTE **)(v10 + 32);
        }
        else if ( v5 )
        {
          memcpy(v14, v3, v5);
          v15 = *(_BYTE **)(v10 + 24);
          v14 = (_BYTE *)(v5 + *(_QWORD *)(v10 + 32));
          *(_QWORD *)(v10 + 32) = v14;
        }
        if ( v14 == v15 )
        {
          sub_CB6200(v10, "\n", 1);
        }
        else
        {
          *v14 = 10;
          ++*(_QWORD *)(v10 + 32);
        }
        return v2;
      }
      while ( v6 != *v7 )
      {
        if ( (_DWORD *)qword_4F819B0 == ++v7 )
          goto LABEL_5;
      }
    }
    v11 = sub_CB72A0(a1, a2);
    v2 = a2;
    v24 = "%2d: ";
    v23 = &unk_49DAA58;
    v25 = dword_4F81904;
    v9 = "DISABLED  ";
    v10 = sub_CB6620(v11, &v23);
    goto LABEL_9;
  }
  v17 = sub_CB72A0(a1, a2);
  v18 = *(void **)(v17 + 32);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0xDu )
  {
    v22 = sub_CB6200(v17, "    DEFAULT   ", 14);
    v20 = *(_BYTE **)(v22 + 32);
    v19 = v22;
  }
  else
  {
    qmemcpy(v18, "    DEFAULT   ", 14);
    v20 = (_BYTE *)(*(_QWORD *)(v17 + 32) + 14LL);
    *(_QWORD *)(v17 + 32) = v20;
  }
  v21 = *(_BYTE **)(v19 + 24);
  if ( v21 - v20 < v5 )
  {
    v19 = sub_CB6200(v19, v3, v5);
    v21 = *(_BYTE **)(v19 + 24);
    v20 = *(_BYTE **)(v19 + 32);
  }
  else if ( v5 )
  {
    memcpy(v20, v3, v5);
    v21 = *(_BYTE **)(v19 + 24);
    v20 = (_BYTE *)(v5 + *(_QWORD *)(v19 + 32));
    *(_QWORD *)(v19 + 32) = v20;
  }
  if ( v20 == v21 )
  {
    sub_CB6200(v19, "\n", 1);
  }
  else
  {
    *v20 = 10;
    ++*(_QWORD *)(v19 + 32);
  }
  return (unsigned int)a2;
}
