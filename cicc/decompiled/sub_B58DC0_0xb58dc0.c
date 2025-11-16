// Function: sub_B58DC0
// Address: 0xb58dc0
//
_QWORD *__fastcall sub_B58DC0(_QWORD *a1, unsigned __int8 **a2)
{
  __int64 v2; // rdx
  _QWORD *result; // rax
  int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // rdx

  v2 = (__int64)*a2;
  result = a1;
  v4 = **a2;
  if ( (unsigned int)(v4 - 1) > 1 )
  {
    if ( (_BYTE)v4 == 4 )
    {
      v5 = *(_QWORD *)(v2 + 136);
      v6 = (v5 + 8LL * *(unsigned int *)(v2 + 144)) | 4;
      *a1 = v5 | 4;
      a1[1] = v6;
    }
    else
    {
      *a1 = 0;
      a1[1] = 0;
    }
  }
  else
  {
    *a1 = v2 & 0xFFFFFFFFFFFFFFFBLL;
    a1[1] = (v2 + 144) & 0xFFFFFFFFFFFFFFFBLL;
  }
  return result;
}
