// Function: sub_80BC40
// Address: 0x80bc40
//
__int64 __fastcall sub_80BC40(char *s, _QWORD *a2)
{
  size_t v2; // rax
  __int64 v3; // rdx
  size_t v4; // rax
  _WORD v6[48]; // [rsp+0h] [rbp-60h] BYREF

  if ( s )
  {
    v2 = strlen(s);
    if ( v2 > 9 )
    {
      v3 = (int)sub_622470(v2, v6);
    }
    else
    {
      v3 = 1;
      v6[0] = (unsigned __int8)(v2 + 48);
    }
    *a2 += v3;
    sub_8238B0(qword_4F18BE0, v6, v3);
    v4 = strlen(s);
    *a2 += v4;
    return sub_8238B0(qword_4F18BE0, s, v4);
  }
  else
  {
    ++*a2;
    v6[0] = 48;
    return sub_8238B0(qword_4F18BE0, v6, 1);
  }
}
