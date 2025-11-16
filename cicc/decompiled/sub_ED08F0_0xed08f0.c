// Function: sub_ED08F0
// Address: 0xed08f0
//
__int64 __fastcall sub_ED08F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  int v3; // ebx
  char *v4; // r15
  unsigned __int64 v5; // r12
  unsigned int v7; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v8; // [rsp+10h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 40);
  v2 = *(_QWORD *)(v1 + 200);
  v8 = *(_QWORD *)(v1 + 208);
  if ( (_BYTE)qword_4F8A848 )
  {
    v3 = qword_4F8A768;
    if ( !(_DWORD)qword_4F8A768 )
      return v2;
  }
  else
  {
    v3 = -1;
  }
  if ( v2 != v2 + v8 )
  {
    v7 = 0;
    v4 = *(char **)(v1 + 200);
    while ( 1 )
    {
      v5 = (unsigned int)(1 - v2 + (_DWORD)v4);
      if ( sub_C80220(*v4, 0) )
      {
        if ( !--v3 )
        {
          if ( v8 >= v5 )
          {
LABEL_10:
            v2 += v5;
            return v2;
          }
LABEL_14:
          v2 += v8;
          return v2;
        }
        v7 = 1 - v2 + (_DWORD)v4;
      }
      if ( (char *)(v2 + v8) == ++v4 )
      {
        v5 = v7;
        if ( v8 >= v7 )
          goto LABEL_10;
        goto LABEL_14;
      }
    }
  }
  return v2;
}
