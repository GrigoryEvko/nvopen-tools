// Function: sub_1446890
// Address: 0x1446890
//
__int64 __fastcall sub_1446890(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 *v10; // rax
  char v11; // dl
  __int64 *v12; // rcx
  unsigned int v13; // edi
  __int64 *v14; // rsi
  __int64 v15[3]; // [rsp+0h] [rbp-40h] BYREF
  char v16; // [rsp+18h] [rbp-28h]

  v2 = *(_QWORD *)(a1 + 112);
  do
  {
    v3 = *(_QWORD *)(v2 - 32);
    if ( !*(_BYTE *)(v2 - 8) )
    {
      v4 = sub_157EBA0(*(_QWORD *)(v2 - 32));
      *(_BYTE *)(v2 - 8) = 1;
      *(_QWORD *)(v2 - 24) = v4;
      *(_DWORD *)(v2 - 16) = 0;
    }
    while ( 1 )
    {
      v5 = sub_157EBA0(v3);
      result = 0;
      if ( v5 )
        result = sub_15F4D60(v5);
      v7 = *(unsigned int *)(v2 - 16);
      if ( (_DWORD)v7 == (_DWORD)result )
        break;
      v8 = *(_QWORD *)(v2 - 24);
      *(_DWORD *)(v2 - 16) = v7 + 1;
      v9 = sub_15F4DF0(v8, v7);
      v10 = *(__int64 **)(a1 + 8);
      if ( *(__int64 **)(a1 + 16) != v10 )
        goto LABEL_8;
      v12 = &v10[*(unsigned int *)(a1 + 28)];
      v13 = *(_DWORD *)(a1 + 28);
      if ( v10 == v12 )
      {
LABEL_19:
        if ( v13 < *(_DWORD *)(a1 + 24) )
        {
          *(_DWORD *)(a1 + 28) = v13 + 1;
          *v12 = v9;
          ++*(_QWORD *)a1;
LABEL_9:
          v15[0] = v9;
          v16 = 0;
          return sub_1446840((__int64 *)(a1 + 104), (__int64)v15);
        }
LABEL_8:
        sub_16CCBA0(a1, v9);
        if ( v11 )
          goto LABEL_9;
      }
      else
      {
        v14 = 0;
        while ( v9 != *v10 )
        {
          if ( *v10 == -2 )
          {
            v14 = v10;
            if ( v10 + 1 == v12 )
              goto LABEL_16;
            ++v10;
          }
          else if ( v12 == ++v10 )
          {
            if ( !v14 )
              goto LABEL_19;
LABEL_16:
            *v14 = v9;
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_9;
          }
        }
      }
    }
    *(_QWORD *)(a1 + 112) -= 32LL;
    v2 = *(_QWORD *)(a1 + 112);
  }
  while ( v2 != *(_QWORD *)(a1 + 104) );
  return result;
}
