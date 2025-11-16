// Function: sub_733C00
// Address: 0x733c00
//
void __fastcall sub_733C00(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx

  v1 = a1[5];
  if ( !v1 )
    goto LABEL_6;
  v2 = *(_QWORD *)(v1 + 32);
  if ( *(_BYTE *)(v1 + 8) )
  {
    sub_733650(a1[5]);
    v3 = *(_QWORD *)(v2 + 48);
    v4 = *(_QWORD *)(v1 + 56);
    if ( v1 != v3 )
      goto LABEL_4;
LABEL_10:
    *(_QWORD *)(v2 + 48) = v4;
    goto LABEL_6;
  }
  v3 = *(_QWORD *)(v2 + 48);
  v4 = *(_QWORD *)(v1 + 56);
  if ( v1 == v3 )
    goto LABEL_10;
  do
  {
LABEL_4:
    v5 = v3;
    v3 = *(_QWORD *)(v3 + 56);
  }
  while ( v1 != v3 );
  *(_QWORD *)(v5 + 56) = v4;
LABEL_6:
  if ( a1[3] )
    sub_733B20(a1);
}
