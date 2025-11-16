// Function: sub_1004820
// Address: 0x1004820
//
__int64 __fastcall sub_1004820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 result; // rax
  unsigned __int64 v7; // r15
  __int64 v8; // rdi
  unsigned int v9; // r14d
  unsigned __int64 v10; // rbx
  __int64 v11; // rax

  v5 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    if ( *(_BYTE *)a2 <= 0x15u )
      return sub_AD5840(a1, (unsigned __int8 *)a2, 0);
    if ( (unsigned __int8)sub_1003090(a3, (unsigned __int8 *)a1) )
      return sub_ACA8A0(*(__int64 ***)(v5 + 24));
  }
  if ( (unsigned __int8)sub_1003090(a3, (unsigned __int8 *)a2) )
    return sub_ACADE0(*(__int64 ***)(v5 + 24));
  if ( *(_BYTE *)a2 == 17 )
  {
    v7 = *(unsigned int *)(v5 + 32);
    v8 = a2 + 24;
    v9 = *(_DWORD *)(a2 + 32);
    if ( *(_BYTE *)(v5 + 8) != 17 )
    {
      if ( v9 > 0x40 )
      {
        v10 = **(_QWORD **)(a2 + 24);
        if ( v9 - (unsigned int)sub_C444A0(v8) > 0x40 || v7 <= v10 )
          return sub_9B7C00(a1, (unsigned int)v10);
      }
      else if ( v7 <= *(_QWORD *)(a2 + 24) )
      {
        goto LABEL_26;
      }
LABEL_10:
      result = sub_9B7920((char *)a1);
      if ( result )
        return result;
      if ( *(_DWORD *)(a2 + 32) > 0x40u )
      {
        v10 = **(_QWORD **)(a2 + 24);
        return sub_9B7C00(a1, (unsigned int)v10);
      }
LABEL_26:
      v10 = *(_QWORD *)(a2 + 24);
      return sub_9B7C00(a1, (unsigned int)v10);
    }
    if ( v9 <= 0x40 )
    {
      if ( *(_QWORD *)(a2 + 24) < v7 )
        goto LABEL_10;
    }
    else if ( v9 - (unsigned int)sub_C444A0(v8) <= 0x40 && **(_QWORD **)(a2 + 24) < v7 )
    {
      goto LABEL_10;
    }
    return sub_ACADE0(*(__int64 ***)(v5 + 24));
  }
  if ( *(_BYTE *)a1 == 91 && (v11 = *(_QWORD *)(a1 - 32), a2 == v11) && v11 )
    return *(_QWORD *)(a1 - 64);
  else
    return sub_9B7920((char *)a1);
}
