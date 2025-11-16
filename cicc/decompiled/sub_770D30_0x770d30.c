// Function: sub_770D30
// Address: 0x770d30
//
void __fastcall sub_770D30(__int64 a1)
{
  __int64 **v1; // rbx
  _QWORD *v2; // r12
  __int64 *v3; // rdx
  __int64 i; // rax

  v1 = *(__int64 ***)(a1 + 72);
  if ( v1 && *v1 )
  {
    v2 = (_QWORD *)(a1 + 96);
    while ( 1 )
    {
      v3 = v1[1];
      if ( !v3 )
        goto LABEL_6;
      if ( (*((_BYTE *)v3 + 89) & 4) != 0 )
        break;
      if ( v3[7] )
      {
LABEL_5:
        sub_686E10(0xCBAu, (FILE *)v1[2], *v3, v2);
LABEL_6:
        v1 = (__int64 **)*v1;
        if ( !*v1 )
          return;
      }
      else
      {
LABEL_12:
        sub_6855B0(0xA85u, (FILE *)v1[2], v2);
        v1 = (__int64 **)*v1;
        if ( !*v1 )
          return;
      }
    }
    for ( i = *(_QWORD *)(v3[5] + 32); (*(_BYTE *)(i + 89) & 4) != 0; i = *(_QWORD *)(*(_QWORD *)(i + 40) + 32LL) )
      ;
    if ( !*(_QWORD *)(i + 56) )
      goto LABEL_12;
    goto LABEL_5;
  }
}
