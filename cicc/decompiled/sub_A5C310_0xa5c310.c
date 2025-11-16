// Function: sub_A5C310
// Address: 0xa5c310
//
__int64 __fastcall sub_A5C310(__int64 a1, __int64 a2, __int64 *a3)
{
  _WORD *v5; // rdx
  __int64 result; // rax
  int v7; // ecx
  __int64 v8; // rbx
  unsigned __int8 *v9; // rsi
  __int64 v10; // r14
  _BYTE *v11; // rax
  _WORD *v12; // rdx
  _DWORD *v13; // rdx
  __int64 v14; // [rsp+18h] [rbp-38h]

  v5 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 1u )
  {
    sub_CB6200(a1, "!{", 2);
  }
  else
  {
    *v5 = 31521;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  result = *(unsigned __int8 *)(a2 - 16);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
  {
    v7 = *(_DWORD *)(a2 - 24);
    if ( v7 )
      goto LABEL_5;
  }
  else
  {
    v7 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    if ( v7 )
    {
LABEL_5:
      v8 = 0;
      v14 = 8LL * (unsigned int)(v7 - 1);
      if ( (*(_BYTE *)(a2 - 16) & 2) == 0 )
        goto LABEL_15;
LABEL_6:
      v9 = *(unsigned __int8 **)(*(_QWORD *)(a2 - 32) + v8);
      if ( v9 )
      {
LABEL_7:
        if ( (unsigned int)*v9 - 1 > 1 )
        {
          sub_A5C090(a1, (__int64)v9, a3);
          result = (*(__int64 (__fastcall **)(__int64 *, unsigned __int8 *))*a3)(a3, v9);
          v12 = *(_WORD **)(a1 + 32);
        }
        else
        {
          v10 = *((_QWORD *)v9 + 17);
          sub_A57EC0(a3[1], *(_QWORD *)(v10 + 8), a1);
          v11 = *(_BYTE **)(a1 + 32);
          if ( (unsigned __int64)v11 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, 32);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v11 + 1;
            *v11 = 32;
          }
          result = (__int64)sub_A5A730(a1, v10, (__int64)a3);
          v12 = *(_WORD **)(a1 + 32);
        }
      }
      else
      {
        while ( 1 )
        {
          v13 = *(_DWORD **)(a1 + 32);
          if ( *(_QWORD *)(a1 + 24) - (_QWORD)v13 <= 3u )
            break;
          *v13 = 1819047278;
          result = *(_QWORD *)(a1 + 32);
          v12 = (_WORD *)(result + 4);
          *(_QWORD *)(a1 + 32) = result + 4;
          if ( v14 == v8 )
            goto LABEL_18;
LABEL_12:
          if ( *(_QWORD *)(a1 + 24) - (_QWORD)v12 <= 1u )
          {
            sub_CB6200(a1, ", ", 2);
          }
          else
          {
            *v12 = 8236;
            *(_QWORD *)(a1 + 32) += 2LL;
          }
          LOBYTE(result) = *(_BYTE *)(a2 - 16);
          v8 += 8;
          if ( (result & 2) != 0 )
            goto LABEL_6;
LABEL_15:
          v9 = *(unsigned __int8 **)(a2 - 16 - 8LL * (((unsigned __int8)result >> 2) & 0xF) + v8);
          if ( v9 )
            goto LABEL_7;
        }
        result = sub_CB6200(a1, "null", 4);
        v12 = *(_WORD **)(a1 + 32);
      }
      if ( v14 != v8 )
        goto LABEL_12;
LABEL_18:
      if ( *(_WORD **)(a1 + 24) != v12 )
        goto LABEL_19;
      return sub_CB6200(a1, "}", 1);
    }
  }
  v12 = *(_WORD **)(a1 + 32);
  if ( *(_WORD **)(a1 + 24) != v12 )
  {
LABEL_19:
    *(_BYTE *)v12 = 125;
    ++*(_QWORD *)(a1 + 32);
    return result;
  }
  return sub_CB6200(a1, "}", 1);
}
