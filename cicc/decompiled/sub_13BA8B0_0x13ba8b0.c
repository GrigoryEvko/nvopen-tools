// Function: sub_13BA8B0
// Address: 0x13ba8b0
//
__int64 *__fastcall sub_13BA8B0(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 *result; // rax
  char v5; // dl
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // rcx
  unsigned int v9; // edi
  _QWORD *v10; // rsi
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 112);
  do
  {
    v3 = *(_QWORD *)(v2 - 24);
    if ( !*(_BYTE *)(v2 - 8) )
    {
      result = *(__int64 **)(v3 + 24);
      *(_BYTE *)(v2 - 8) = 1;
      *(_QWORD *)(v2 - 16) = result;
      goto LABEL_6;
    }
    while ( 1 )
    {
      result = *(__int64 **)(v2 - 16);
LABEL_6:
      if ( *(__int64 **)(v3 + 32) == result )
        break;
      *(_QWORD *)(v2 - 16) = result + 1;
      v6 = *result;
      v7 = *(_QWORD **)(a1 + 8);
      if ( *(_QWORD **)(a1 + 16) != v7 )
        goto LABEL_4;
      v8 = &v7[*(unsigned int *)(a1 + 28)];
      v9 = *(_DWORD *)(a1 + 28);
      if ( v7 == v8 )
      {
LABEL_18:
        if ( v9 < *(_DWORD *)(a1 + 24) )
        {
          *(_DWORD *)(a1 + 28) = v9 + 1;
          *v8 = v6;
          ++*(_QWORD *)a1;
LABEL_15:
          v11 = v6;
          v12 = 0;
          return (__int64 *)sub_13B8390((unsigned __int64 *)(a1 + 104), (__int64)&v11);
        }
LABEL_4:
        sub_16CCBA0(a1, v6);
        if ( v5 )
          goto LABEL_15;
      }
      else
      {
        v10 = 0;
        while ( v6 != *v7 )
        {
          if ( *v7 == -2 )
          {
            v10 = v7;
            if ( v7 + 1 == v8 )
              goto LABEL_14;
            ++v7;
          }
          else if ( v8 == ++v7 )
          {
            if ( !v10 )
              goto LABEL_18;
LABEL_14:
            *v10 = v6;
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_15;
          }
        }
      }
    }
    *(_QWORD *)(a1 + 112) -= 24LL;
    v2 = *(_QWORD *)(a1 + 112);
  }
  while ( v2 != *(_QWORD *)(a1 + 104) );
  return result;
}
