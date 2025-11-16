// Function: sub_E4D060
// Address: 0xe4d060
//
__int64 __fastcall sub_E4D060(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _WORD *v6; // rdx
  __int64 result; // rax
  char *v8; // r15
  char *v9; // r12
  char v10; // al
  __int64 v11; // rax
  void *v13; // [rsp+20h] [rbp-50h] BYREF
  const char *v14; // [rsp+28h] [rbp-48h]
  char v15; // [rsp+30h] [rbp-40h]

  v6 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v6 <= 0xCu )
  {
    sub_CB6200(a1, "\t.cfi_escape ", 0xDu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_escape ", 13);
    *(_QWORD *)(a1 + 32) += 13LL;
  }
  result = a3;
  if ( a3 )
  {
    if ( a3 != 1 )
    {
      v8 = a2;
      v9 = &a2[a3 - 1];
      do
      {
        while ( 1 )
        {
          v10 = *v8;
          v14 = "0x%02x";
          v13 = &unk_49DD0D8;
          v15 = v10;
          v11 = sub_CB6620(a1, (__int64)&v13, (__int64)v6, (__int64)&unk_49DD0D8, a5, a6);
          v6 = *(_WORD **)(v11 + 32);
          if ( *(_QWORD *)(v11 + 24) - (_QWORD)v6 <= 1u )
            break;
          ++v8;
          *v6 = 8236;
          *(_QWORD *)(v11 + 32) += 2LL;
          if ( v9 == v8 )
            goto LABEL_9;
        }
        ++v8;
        sub_CB6200(v11, (unsigned __int8 *)", ", 2u);
      }
      while ( v9 != v8 );
    }
LABEL_9:
    v14 = "0x%02x";
    v15 = a2[a3 - 1];
    v13 = &unk_49DD0D8;
    return sub_CB6620(a1, (__int64)&v13, (__int64)&unk_49DD0D8, (__int64)&unk_49DD0C8, a5, a6);
  }
  return result;
}
