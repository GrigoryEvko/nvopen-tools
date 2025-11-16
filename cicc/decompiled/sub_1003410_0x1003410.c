// Function: sub_1003410
// Address: 0x1003410
//
__int64 __fastcall sub_1003410(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // r13
  unsigned __int64 v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rax
  int v9; // r13d
  unsigned int v10; // r14d
  __int64 v11; // rax

  if ( *(_BYTE *)a1 > 0x15u )
    return 0;
  if ( (unsigned __int8)sub_1003090(a2, (unsigned __int8 *)a1) )
    return 1;
  v3 = *(_BYTE *)a1;
  v4 = a1 + 24;
  if ( *(_BYTE *)a1 == 17 )
    goto LABEL_5;
  v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17;
  if ( (unsigned int)v7 <= 1 )
  {
    v8 = sub_AD7630(a1, 0, v7);
    if ( !v8 || *v8 != 17 )
    {
LABEL_8:
      v3 = *(_BYTE *)a1;
      goto LABEL_9;
    }
    v4 = (__int64)(v8 + 24);
LABEL_5:
    v5 = *(unsigned int *)(v4 + 8);
    if ( (unsigned int)v5 > 0x40 )
    {
      if ( (unsigned int)v5 - (unsigned int)sub_C444A0(v4) > 0x40 )
        return 1;
      v6 = **(_QWORD ***)v4;
    }
    else
    {
      v6 = *(_QWORD **)v4;
    }
    if ( v5 <= (unsigned __int64)v6 )
      return 1;
    goto LABEL_8;
  }
LABEL_9:
  if ( v3 == 16 || v3 == 11 )
  {
    v9 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
    if ( !v9 )
      return 1;
    v10 = 0;
    while ( 1 )
    {
      v11 = sub_AD69F0((unsigned __int8 *)a1, v10);
      if ( !(unsigned __int8)sub_1003410(v11, a2) )
        break;
      if ( ++v10 == v9 )
        return 1;
    }
  }
  return 0;
}
