// Function: sub_819EB0
// Address: 0x819eb0
//
_BYTE *__fastcall sub_819EB0(void *src, __int64 *a2)
{
  size_t v2; // rax
  size_t v3; // r13
  __int64 v4; // r15
  _BYTE *v5; // rax
  _BYTE *v6; // r14
  __int64 v8; // rax

  if ( src && (v2 = strlen((const char *)src), (v3 = v2) != 0) )
  {
    v4 = v2 + 5;
    v8 = sub_823970(v2 + 5);
    *(_BYTE *)v8 = 1;
    v6 = (_BYTE *)v8;
    *(_WORD *)(v8 + 1) = v3;
    *(_BYTE *)(v8 + 3) = BYTE2(v3);
    v5 = (char *)memcpy((void *)(v8 + 4), src, v3) + v3;
  }
  else
  {
    v4 = 1;
    v5 = (_BYTE *)sub_823970(1);
    v6 = v5;
  }
  *v5 = 0;
  if ( a2 )
    *a2 = v4;
  return v6;
}
