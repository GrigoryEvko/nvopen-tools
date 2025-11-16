// Function: sub_7D78E0
// Address: 0x7d78e0
//
_QWORD *__fastcall sub_7D78E0(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v3; // rax
  __int64 v4; // rdi
  __int64 *v5; // rax

  v1 = 0;
  do
  {
    while ( *(_BYTE *)(a1 + 140) == 12 )
      a1 = *(_QWORD *)(a1 + 160);
    if ( (*(_BYTE *)(a1 + 169) & 2) != 0 )
    {
      v3 = sub_72D900(a1);
      v4 = v3[6];
      if ( v4 )
        v5 = sub_73E830(v4);
      else
        v5 = (__int64 *)sub_7E8090(v3[2], 1);
      if ( v1 )
      {
LABEL_8:
        v1[2] = v5;
        v1 = sub_73DBF0(0x29u, *v5, (__int64)v1);
        goto LABEL_9;
      }
    }
    else
    {
      v5 = sub_73A8E0(*(_QWORD *)(a1 + 176), 5u);
      if ( v1 )
        goto LABEL_8;
    }
    v1 = v5;
LABEL_9:
    a1 = sub_8D4050(a1);
  }
  while ( (unsigned int)sub_8D3410(a1) );
  return v1;
}
