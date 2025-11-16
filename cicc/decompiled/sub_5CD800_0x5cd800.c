// Function: sub_5CD800
// Address: 0x5cd800
//
__int64 __fastcall sub_5CD800(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  const char *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // rax

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
  v6 = *(_QWORD *)(v5 + 184);
  if ( *(_BYTE *)(a1 + 9) == 3 )
  {
    if ( a3 == 11 )
    {
      sub_5CCAE0(8u, a1);
      if ( !*(_BYTE *)(a1 + 8) )
        return a2;
      v10 = sub_724840(unk_4F073B8, v6);
      goto LABEL_9;
    }
    v8 = *(const char **)(a2 + 224);
    if ( v8 && strcmp(v8, *(const char **)(v5 + 184)) )
    {
      sub_684AA0(7, 654, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
  }
  if ( !*(_BYTE *)(a1 + 8) )
    return a2;
  v9 = sub_724840(unk_4F073B8, v6);
  v10 = v9;
  if ( a3 == 7 )
  {
    *(_QWORD *)(a2 + 224) = v9;
    return a2;
  }
LABEL_9:
  v11 = *(__int64 **)(a2 + 256);
  if ( !v11 )
    v11 = (__int64 *)sub_726210(a2);
  *v11 = v10;
  return a2;
}
