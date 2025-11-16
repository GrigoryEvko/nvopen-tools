// Function: sub_1F56CD0
// Address: 0x1f56cd0
//
__int64 __fastcall sub_1F56CD0(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  unsigned int v7; // esi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 *v12; // rax
  char v13; // dl
  __int64 *v14; // rcx
  unsigned int v15; // r8d
  __int64 *v16; // rsi
  __int64 v17[3]; // [rsp+0h] [rbp-40h] BYREF
  char v18; // [rsp+18h] [rbp-28h]

  v2 = a1[2];
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
      v7 = *(_DWORD *)(v2 - 16);
      if ( v7 == (_DWORD)result )
        break;
      v8 = *(_QWORD *)(v2 - 24);
      *(_DWORD *)(v2 - 16) = v7 + 1;
      v9 = sub_15F4DF0(v8, v7);
      v10 = *a1;
      v11 = v9;
      v12 = *(__int64 **)(*a1 + 8);
      if ( *(__int64 **)(*a1 + 16) != v12 )
        goto LABEL_8;
      v14 = &v12[*(unsigned int *)(v10 + 28)];
      v15 = *(_DWORD *)(v10 + 28);
      if ( v12 == v14 )
      {
LABEL_19:
        if ( v15 < *(_DWORD *)(v10 + 24) )
        {
          *(_DWORD *)(v10 + 28) = v15 + 1;
          *v14 = v11;
          ++*(_QWORD *)v10;
LABEL_9:
          v17[0] = v11;
          v18 = 0;
          return sub_144A690(a1 + 1, (__int64)v17);
        }
LABEL_8:
        sub_16CCBA0(v10, v11);
        if ( v13 )
          goto LABEL_9;
      }
      else
      {
        v16 = 0;
        while ( v11 != *v12 )
        {
          if ( *v12 == -2 )
          {
            v16 = v12;
            if ( v12 + 1 == v14 )
              goto LABEL_16;
            ++v12;
          }
          else if ( v14 == ++v12 )
          {
            if ( !v16 )
              goto LABEL_19;
LABEL_16:
            *v16 = v11;
            --*(_DWORD *)(v10 + 32);
            ++*(_QWORD *)v10;
            goto LABEL_9;
          }
        }
      }
    }
    a1[2] -= 32;
    v2 = a1[2];
  }
  while ( v2 != a1[1] );
  return result;
}
