// Function: sub_6F68A0
// Address: 0x6f68a0
//
__int64 __fastcall sub_6F68A0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // dl
  unsigned int v4; // r8d
  __int64 i; // rax
  unsigned int v7; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v7 = 0;
  sub_6F6860(a1, 0, v8, &v7);
  v2 = v8[0];
  if ( v8[0] )
  {
    v3 = *(_BYTE *)(v8[0] + 80);
    if ( v3 == 16 )
    {
      v2 = **(_QWORD **)(v8[0] + 88);
      v3 = *(_BYTE *)(v2 + 80);
      if ( v3 != 24 )
      {
LABEL_4:
        v4 = 0;
        if ( v3 != 10 )
          return v4;
        goto LABEL_6;
      }
    }
    else if ( v3 != 24 )
    {
      goto LABEL_4;
    }
    v2 = *(_QWORD *)(v2 + 88);
    v4 = 0;
    if ( *(_BYTE *)(v2 + 80) != 10 )
      return v4;
LABEL_6:
    for ( i = *(_QWORD *)(*(_QWORD *)(v2 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v4 = 0;
    if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      return v4;
    goto LABEL_9;
  }
  v4 = v7;
  if ( !v7 )
    return v4;
LABEL_9:
  a1[1].m128i_i8[2] &= ~1u;
  sub_82F8F0(a2, (*(_BYTE *)(a2 + 18) & 2) != 0, a1);
  return 1;
}
