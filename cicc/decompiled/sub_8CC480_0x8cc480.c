// Function: sub_8CC480
// Address: 0x8cc480
//
__int64 *__fastcall sub_8CC480(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 *result; // rax
  __int64 *v5; // r14
  __int64 v6; // rsi
  _QWORD *v7; // rdx
  __int64 v8; // rdx

  v1 = a1;
  v2 = *(_QWORD **)a1;
  if ( (*(_BYTE *)(a1 + 124) & 1) == 0 )
  {
    if ( qword_4D049B8 != v2 )
      goto LABEL_3;
LABEL_18:
    v6 = v1;
    v7 = *(_QWORD **)(qword_4D03FD0[1] + 168LL);
    if ( (_QWORD *)v1 == v7 )
      return sub_8C7090(28, v6);
    return sub_8CBB20(0x1Cu, v6, v7);
  }
  a1 = sub_735B70(a1);
  if ( qword_4D049B8 == v2 )
    goto LABEL_18;
LABEL_3:
  if ( qword_4D04998 == v2 )
  {
    v6 = v1;
    v7 = *(_QWORD **)(*(_QWORD *)(qword_4D03FD0[1] + 168LL) + 112LL);
    if ( (_QWORD *)v1 == v7 )
      return sub_8C7090(28, v6);
    return sub_8CBB20(0x1Cu, v6, v7);
  }
  if ( (unsigned int)sub_736990(a1) )
  {
    v6 = v1;
    return sub_8C7090(28, v6);
  }
  v3 = *(_QWORD *)(*v2 + 32LL);
  result = (__int64 *)sub_880F80((__int64)v2);
  v5 = result;
  if ( !*(_QWORD *)(v1 + 32) )
    result = sub_8C7090(28, v1);
  while ( v3 )
  {
    if ( *(_DWORD *)(v3 + 40) != -1 )
    {
      result = (__int64 *)sub_880F80(v3);
      if ( v5 != result )
      {
        result = (__int64 *)sub_8C7F70(v3, (__int64)v2);
        if ( (_DWORD)result )
        {
          if ( (unsigned int)sub_8C6B40(v3) )
          {
            result = *(__int64 **)(v1 + 32);
            if ( !result || *result == v1 && result[1] != v1 )
            {
              if ( *(_BYTE *)(v3 + 80) == 23
                && (v8 = *(_QWORD *)(v3 + 88), ((*(_BYTE *)(v1 + 124) ^ *(_BYTE *)(v8 + 124)) & 1) == 0) )
              {
                result = sub_8CBB20(0x1Cu, v1, (_QWORD *)v8);
              }
              else
              {
                result = (__int64 *)sub_8C6700((__int64 *)v1, (unsigned int *)(v3 + 48), 0x42Au, 0x425u);
              }
            }
          }
          else
          {
            result = (__int64 *)sub_87D520(v3);
            if ( result && (*(_BYTE *)(result - 1) & 2) == 0 )
              *((_BYTE *)result + 90) |= 8u;
          }
        }
      }
    }
    v3 = *(_QWORD *)(v3 + 8);
  }
  return result;
}
