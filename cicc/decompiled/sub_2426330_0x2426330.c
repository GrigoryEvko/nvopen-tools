// Function: sub_2426330
// Address: 0x2426330
//
__int64 __fastcall sub_2426330(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v2; // al
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v9; // r12

  v1 = a1 - 16;
  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) == 0 )
  {
    v4 = *(_QWORD *)(v1 - 8LL * ((v2 >> 2) & 0xF) + 24);
    if ( !v4 )
      goto LABEL_9;
    goto LABEL_3;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 24LL);
  if ( v4 )
  {
LABEL_3:
    sub_B91420(v4);
    v2 = *(_BYTE *)(a1 - 16);
    if ( v5 )
    {
      if ( (v2 & 2) != 0 )
        v6 = *(_QWORD *)(a1 - 32);
      else
        v6 = v1 - 8LL * ((v2 >> 2) & 0xF);
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 )
        return sub_B91420(v7);
      return v7;
    }
    if ( (v2 & 2) != 0 )
      goto LABEL_14;
LABEL_9:
    v9 = v1 - 8LL * ((v2 >> 2) & 0xF);
    goto LABEL_10;
  }
LABEL_14:
  v9 = *(_QWORD *)(a1 - 32);
LABEL_10:
  v7 = *(_QWORD *)(v9 + 16);
  if ( v7 )
    return sub_B91420(v7);
  return v7;
}
