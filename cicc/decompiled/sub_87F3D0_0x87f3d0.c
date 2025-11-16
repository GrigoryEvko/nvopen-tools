// Function: sub_87F3D0
// Address: 0x87f3d0
//
_QWORD *__fastcall sub_87F3D0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // al
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // r12
  __int64 v7; // rdx

  v1 = *(_QWORD *)(a1 + 88);
  if ( (*(_BYTE *)(v1 + 265) & 1) == 0 )
  {
    v2 = *(_BYTE *)(v1 + 264);
    if ( v2 <= 0xAu )
    {
      if ( v2 > 8u )
      {
        v6 = sub_87EBB0(4u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
        goto LABEL_5;
      }
    }
    else if ( v2 == 11 )
    {
      v6 = sub_87EBB0(5u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
LABEL_5:
      *(_QWORD *)(v6[12] + 72LL) = a1;
      goto LABEL_6;
    }
    sub_721090();
  }
  v6 = sub_87EBB0(3u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
LABEL_6:
  *((_DWORD *)v6 + 10) = *(_DWORD *)(a1 + 40);
  v7 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    sub_877E20((__int64)v6, 0, v7, v3, v4, v5);
    return v6;
  }
  else
  {
    if ( v7 )
      sub_877E90((__int64)v6, 0, v7);
    return v6;
  }
}
