// Function: sub_36E5B70
// Address: 0x36e5b70
//
__int64 __fastcall sub_36E5B70(__int64 *a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  _QWORD *v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v11; // r13

  v7 = *(_QWORD **)(a2 + 40);
  v8 = *(_QWORD *)(*v7 + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( (_DWORD)v9 == 9157 )
    goto LABEL_25;
  if ( (unsigned int)v9 > 0x23C5 )
  {
    if ( (_DWORD)v9 == 9578 )
      goto LABEL_34;
    if ( (unsigned int)v9 > 0x256A )
    {
      if ( (_DWORD)v9 == 10579 )
      {
        sub_36DD6A0((__int64)a1, a2);
        LODWORD(a6) = 1;
        return (unsigned int)a6;
      }
      a6 = 0;
      if ( (unsigned int)v9 > 0x2953 )
        return (unsigned int)a6;
      if ( (unsigned int)v9 > 0x256C )
      {
        if ( (_DWORD)v9 != 9597 )
          return (unsigned int)a6;
        return sub_36E41D0((__int64)a1, a2, (__int64)v9, (__int64)v7, a6, a7);
      }
LABEL_25:
      sub_36E5710((__int64)a1, a2, a3);
      LODWORD(a6) = 1;
      return (unsigned int)a6;
    }
    if ( (_DWORD)v9 == 9576 )
    {
LABEL_34:
      sub_36E5180((__int64)a1, a2, a3);
      LODWORD(a6) = 1;
      return (unsigned int)a6;
    }
    a6 = 0;
    if ( (unsigned int)v9 > 0x2568 )
      return (unsigned int)a6;
    if ( (_DWORD)v9 == 9160 )
      goto LABEL_25;
    if ( (_DWORD)v9 != 9175 )
      return (unsigned int)a6;
    return sub_36E41D0((__int64)a1, a2, (__int64)v9, (__int64)v7, a6, a7);
  }
  if ( (_DWORD)v9 == 8285 )
  {
    v11 = v7[5];
    sub_34158F0(a1[8], a2, v11, (__int64)v7, a6, a7);
    sub_3421DB0(v11);
    sub_33ECEA0((const __m128i *)a1[8], a2);
    LODWORD(a6) = 1;
    return (unsigned int)a6;
  }
  if ( (unsigned int)v9 <= 0x205D )
  {
    if ( (_DWORD)v9 == 8165 )
    {
      sub_36E54E0(a1, a2);
      LODWORD(a6) = 1;
      return (unsigned int)a6;
    }
    a6 = 0;
    if ( (unsigned int)v9 > 0x1FE5 )
      return (unsigned int)a6;
    if ( (unsigned int)v9 <= 0x1FC7 )
    {
      if ( (unsigned int)v9 <= 0x1FC5 )
        return (unsigned int)a6;
      goto LABEL_25;
    }
    if ( (_DWORD)v9 != 8164 )
      return (unsigned int)a6;
    return sub_36E41D0((__int64)a1, a2, (__int64)v9, (__int64)v7, a6, a7);
  }
  if ( (unsigned int)v9 <= 0x21CB )
  {
    if ( (unsigned int)v9 > 0x21C9 )
      goto LABEL_25;
  }
  else if ( (_DWORD)v9 == 9004 )
  {
    return sub_36E41D0((__int64)a1, a2, (__int64)v9, (__int64)v7, a6, a7);
  }
  return 0;
}
