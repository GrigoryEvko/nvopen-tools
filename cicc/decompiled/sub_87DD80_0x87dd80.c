// Function: sub_87DD80
// Address: 0x87dd80
//
_QWORD *sub_87DD80()
{
  _QWORD *result; // rax
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // rsi
  _QWORD *v4; // rcx
  unsigned int v5; // edx
  unsigned int v6[5]; // [rsp+Ch] [rbp-14h] BYREF

  result = qword_4F04C68;
  v1 = qword_4F04C68[0] + 776LL * (int)dword_4F04C40;
  v2 = *(_QWORD **)(v1 + 456);
  if ( v2 )
  {
    sub_7BE840(v6, 0);
    v3 = 0;
    v4 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        result = v2;
        v2 = (_QWORD *)*v2;
        v5 = *((_DWORD *)result + 10);
        *result = 0;
        if ( v5 < dword_4F06650[0] || v5 >= v6[0] )
          break;
        *result = qword_4F60008;
        qword_4F60008 = (__int64)result;
LABEL_5:
        if ( !v2 )
          goto LABEL_11;
      }
      if ( !v4 )
        v4 = result;
      if ( !v3 )
      {
        v3 = result;
        goto LABEL_5;
      }
      *v3 = result;
      v3 = result;
      if ( !v2 )
      {
LABEL_11:
        *(_QWORD *)(v1 + 456) = v4;
        *(_QWORD *)(v1 + 464) = v3;
        return result;
      }
    }
  }
  return result;
}
