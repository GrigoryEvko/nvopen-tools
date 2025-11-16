// Function: sub_16B88A0
// Address: 0x16b88a0
//
__int64 *__fastcall sub_16B88A0(__int64 a1)
{
  const char **v1; // r14
  __int64 v2; // rdx
  __int64 *result; // rax
  __int64 *v4; // r12
  __int64 v5; // rdx
  __int64 *v6; // rbx
  __int64 v7; // rax

  if ( !qword_4FA01E0 )
    sub_16C1EA0(&qword_4FA01E0, sub_16B89A0, sub_16B0D50);
  v1 = (const char **)qword_4FA01E0;
  v2 = *(unsigned int *)(a1 + 108);
  if ( (_DWORD)v2 == *(_DWORD *)(a1 + 112) )
  {
    v7 = sub_16B4B80((__int64)&unk_4FA0190);
    result = (__int64 *)sub_16B85B0(v1, a1, v7);
  }
  else
  {
    result = *(__int64 **)(a1 + 96);
    if ( result != *(__int64 **)(a1 + 88) )
      v2 = *(unsigned int *)(a1 + 104);
    v4 = &result[v2];
    if ( result != v4 )
    {
      while ( 1 )
      {
        v5 = *result;
        v6 = result;
        if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++result )
          goto LABEL_9;
      }
      if ( result != v4 )
      {
        do
        {
          sub_16B85B0(v1, a1, v5);
          result = v6 + 1;
          if ( v6 + 1 == v4 )
            break;
          v5 = *result;
          for ( ++v6; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v6 = result )
          {
            if ( v4 == ++result )
              goto LABEL_9;
            v5 = *result;
          }
        }
        while ( v4 != v6 );
      }
    }
  }
LABEL_9:
  *(_BYTE *)(a1 + 152) = 1;
  return result;
}
