// Function: sub_1DCCCF0
// Address: 0x1dcccf0
//
__int64 __fastcall sub_1DCCCF0(char *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 i; // rbx
  unsigned __int8 v7; // dl
  int v8; // esi
  char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  char *v12; // rdi
  __int64 v13; // rcx

  result = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)result )
  {
    v4 = 40 * result;
    for ( i = 0; v4 != i; i += 40 )
    {
      result = i + *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)result )
        continue;
      v7 = *(_BYTE *)(result + 3);
      if ( (((v7 & 0x40) != 0) & ((v7 >> 4) ^ 1)) == 0 )
        continue;
      v8 = *(_DWORD *)(result + 8);
      *(_BYTE *)(result + 3) = v7 & 0xBF;
      if ( v8 >= 0 )
        continue;
      v9 = sub_1DCC790(a1, v8);
      v10 = *((_QWORD *)v9 + 5);
      v11 = *((_QWORD *)v9 + 4);
      v12 = v9;
      result = (v10 - v11) >> 5;
      v13 = (v10 - v11) >> 3;
      if ( result > 0 )
      {
        result = v11 + 32 * result;
        while ( a2 != *(_QWORD *)v11 )
        {
          if ( a2 == *(_QWORD *)(v11 + 8) )
          {
            v11 += 8;
            break;
          }
          if ( a2 == *(_QWORD *)(v11 + 16) )
          {
            v11 += 16;
            break;
          }
          if ( a2 == *(_QWORD *)(v11 + 24) )
          {
            v11 += 24;
            break;
          }
          v11 += 32;
          if ( result == v11 )
          {
            v13 = (v10 - v11) >> 3;
            goto LABEL_19;
          }
        }
LABEL_15:
        if ( v10 != v11 )
          result = (__int64)sub_1DCBB50((__int64)(v12 + 32), (_BYTE *)v11);
        continue;
      }
LABEL_19:
      if ( v13 != 2 )
      {
        if ( v13 != 3 )
        {
          if ( v13 != 1 )
            continue;
          goto LABEL_22;
        }
        if ( a2 == *(_QWORD *)v11 )
          goto LABEL_15;
        v11 += 8;
      }
      if ( a2 == *(_QWORD *)v11 )
        goto LABEL_15;
      v11 += 8;
LABEL_22:
      if ( a2 == *(_QWORD *)v11 )
        goto LABEL_15;
    }
  }
  return result;
}
