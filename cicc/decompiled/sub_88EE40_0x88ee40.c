// Function: sub_88EE40
// Address: 0x88ee40
//
__int64 __fastcall sub_88EE40(__int64 a1)
{
  __int64 v2; // r13
  __int64 i; // rdi
  _BYTE *v4; // rbx
  __int64 result; // rax
  bool v6; // zf

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL) + 128LL);
  if ( (unsigned int)sub_8DBE70(v2) )
  {
    if ( !dword_4D04964 )
      goto LABEL_9;
    for ( i = v2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (unsigned int)sub_8D3F00(i) )
LABEL_9:
      v2 = 0;
  }
  v4 = sub_724D80(0);
  result = sub_6D6050(v2, (__int64)v4, 0, 0);
  v6 = v4[173] == 12;
  *((_QWORD *)v4 + 18) = 0;
  if ( v6 )
  {
    *(_WORD *)(a1 + 56) |= 0x202u;
LABEL_8:
    *(_QWORD *)(a1 + 80) = v4;
    return result;
  }
  result = sub_72A990((__int64)v4);
  if ( !(_DWORD)result )
    goto LABEL_8;
  sub_6851C0(0x1DBu, dword_4F07508);
  result = sub_72C970((__int64)v4);
  *(_QWORD *)(a1 + 80) = v4;
  return result;
}
