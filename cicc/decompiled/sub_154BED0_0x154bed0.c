// Function: sub_154BED0
// Address: 0x154bed0
//
__int64 __fastcall sub_154BED0(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v6; // r13
  __int64 v7; // rax

  v1 = a1;
  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 17 )
  {
    v1 = *(_QWORD *)(a1 + 24);
    goto LABEL_3;
  }
  if ( v2 > 0x17u )
  {
    v4 = *(_QWORD *)(a1 + 40);
    if ( v4 )
    {
      v1 = *(_QWORD *)(v4 + 56);
      goto LABEL_3;
    }
    return v4;
  }
  if ( v2 == 18 )
  {
    v1 = *(_QWORD *)(a1 + 56);
    goto LABEL_3;
  }
  if ( v2 != 3 && v2 != 1 && v2 != 2 )
  {
    if ( v2 )
      return 0;
LABEL_3:
    v3 = sub_22077B0(272);
    v4 = v3;
    if ( v3 )
      sub_154BD90(v3, v1, 0);
    return v4;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = sub_22077B0(272);
  v4 = v7;
  if ( !v7 )
    return v4;
  sub_154BB30(v7, v6, 0);
  return v4;
}
