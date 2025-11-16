// Function: sub_B129C0
// Address: 0xb129c0
//
_QWORD *__fastcall sub_B129C0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  _QWORD *result; // rax
  int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // rdx

  v2 = *(_QWORD *)(a2 + 40);
  result = a1;
  if ( !v2 )
    goto LABEL_6;
  v4 = *(unsigned __int8 *)v2;
  if ( (unsigned int)(v4 - 1) <= 1 )
  {
    *a1 = v2 & 0xFFFFFFFFFFFFFFFBLL;
    a1[1] = (v2 + 144) & 0xFFFFFFFFFFFFFFFBLL;
    return result;
  }
  if ( (_BYTE)v4 == 4 )
  {
    v5 = *(_QWORD *)(v2 + 136);
    v6 = (v5 + 8LL * *(unsigned int *)(v2 + 144)) | 4;
    *a1 = v5 | 4;
    a1[1] = v6;
  }
  else
  {
LABEL_6:
    *a1 = 0;
    a1[1] = 0;
  }
  return result;
}
