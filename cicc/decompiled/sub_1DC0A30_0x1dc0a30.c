// Function: sub_1DC0A30
// Address: 0x1dc0a30
//
__int64 *__fastcall sub_1DC0A30(unsigned __int64 *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // r14
  __int64 *result; // rax
  char v5; // dl
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 *v9; // rcx
  unsigned int v10; // r8d
  __int64 *v11; // rsi
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]

  v2 = a1[2];
  do
  {
    v3 = *(_QWORD *)(v2 - 24);
    if ( !*(_BYTE *)(v2 - 8) )
    {
      result = *(__int64 **)(v3 + 88);
      *(_BYTE *)(v2 - 8) = 1;
      *(_QWORD *)(v2 - 16) = result;
      goto LABEL_6;
    }
    while ( 1 )
    {
      result = *(__int64 **)(v2 - 16);
LABEL_6:
      if ( *(__int64 **)(v3 + 96) == result )
        break;
      *(_QWORD *)(v2 - 16) = result + 1;
      v6 = *a1;
      v7 = *result;
      v8 = *(__int64 **)(*a1 + 8);
      if ( *(__int64 **)(*a1 + 16) != v8 )
        goto LABEL_4;
      v9 = &v8[*(unsigned int *)(v6 + 28)];
      v10 = *(_DWORD *)(v6 + 28);
      if ( v8 == v9 )
      {
LABEL_18:
        if ( v10 < *(_DWORD *)(v6 + 24) )
        {
          *(_DWORD *)(v6 + 28) = v10 + 1;
          *v9 = v7;
          ++*(_QWORD *)v6;
LABEL_15:
          v12 = v7;
          v13 = 0;
          return (__int64 *)sub_1BFDD10(a1 + 1, (__int64)&v12);
        }
LABEL_4:
        sub_16CCBA0(v6, v7);
        if ( v5 )
          goto LABEL_15;
      }
      else
      {
        v11 = 0;
        while ( v7 != *v8 )
        {
          if ( *v8 == -2 )
          {
            v11 = v8;
            if ( v8 + 1 == v9 )
              goto LABEL_14;
            ++v8;
          }
          else if ( v9 == ++v8 )
          {
            if ( !v11 )
              goto LABEL_18;
LABEL_14:
            *v11 = v7;
            --*(_DWORD *)(v6 + 32);
            ++*(_QWORD *)v6;
            goto LABEL_15;
          }
        }
      }
    }
    a1[2] -= 24LL;
    v2 = a1[2];
  }
  while ( v2 != a1[1] );
  return result;
}
