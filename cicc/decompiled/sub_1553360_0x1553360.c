// Function: sub_1553360
// Address: 0x1553360
//
__int64 __fastcall sub_1553360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _WORD *v8; // rdx
  __int64 result; // rax
  __int64 v10; // rbx
  unsigned __int8 *v11; // rsi
  _BYTE *v12; // rax
  __int64 v13; // r10
  _WORD *v14; // rdx
  _DWORD *v15; // rdx
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v8 = *(_WORD **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 1u )
  {
    sub_16E7EE0(a1, "!{", 2);
    result = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)result )
    {
LABEL_3:
      v10 = 0;
      v18 = (unsigned int)(result - 1);
      v11 = *(unsigned __int8 **)(a2 - 8 * result);
      if ( !v11 )
        goto LABEL_12;
      while ( 1 )
      {
        if ( (unsigned int)*v11 - 1 > 1 )
        {
          result = (__int64)sub_154F770(a1, v11, a3, a4, a5);
          v14 = *(_WORD **)(a1 + 24);
        }
        else
        {
          v16 = (__int64 *)*((_QWORD *)v11 + 17);
          sub_154DAA0(a3, *v16, a1);
          v12 = *(_BYTE **)(a1 + 24);
          v13 = (__int64)v16;
          if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 16) )
          {
            sub_16E7DE0(a1, 32);
            v13 = (__int64)v16;
          }
          else
          {
            *(_QWORD *)(a1 + 24) = v12 + 1;
            *v12 = 32;
          }
          result = (__int64)sub_1550E20(a1, v13, a3, a4, a5);
          v14 = *(_WORD **)(a1 + 24);
        }
LABEL_8:
        if ( v18 == v10 )
          break;
        while ( 1 )
        {
          if ( *(_QWORD *)(a1 + 16) - (_QWORD)v14 <= 1u )
          {
            sub_16E7EE0(a1, ", ", 2);
          }
          else
          {
            *v14 = 8236;
            *(_QWORD *)(a1 + 24) += 2LL;
          }
          v11 = *(unsigned __int8 **)(a2 + 8 * (++v10 - *(unsigned int *)(a2 + 8)));
          if ( v11 )
            break;
LABEL_12:
          v15 = *(_DWORD **)(a1 + 24);
          if ( *(_QWORD *)(a1 + 16) - (_QWORD)v15 <= 3u )
          {
            result = sub_16E7EE0(a1, "null", 4);
            v14 = *(_WORD **)(a1 + 24);
            goto LABEL_8;
          }
          *v15 = 1819047278;
          result = *(_QWORD *)(a1 + 24);
          v14 = (_WORD *)(result + 4);
          *(_QWORD *)(a1 + 24) = result + 4;
          if ( v18 == v10 )
            goto LABEL_14;
        }
      }
LABEL_14:
      if ( v14 != *(_WORD **)(a1 + 16) )
        goto LABEL_15;
      return sub_16E7EE0(a1, "}", 1);
    }
  }
  else
  {
    *v8 = 31521;
    *(_QWORD *)(a1 + 24) += 2LL;
    result = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)result )
      goto LABEL_3;
  }
  v14 = *(_WORD **)(a1 + 24);
  if ( v14 != *(_WORD **)(a1 + 16) )
  {
LABEL_15:
    *(_BYTE *)v14 = 125;
    ++*(_QWORD *)(a1 + 24);
    return result;
  }
  return sub_16E7EE0(a1, "}", 1);
}
