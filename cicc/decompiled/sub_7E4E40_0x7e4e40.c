// Function: sub_7E4E40
// Address: 0x7e4e40
//
void __fastcall sub_7E4E40(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 i; // rbx
  _QWORD *j; // rbx
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = a1[13];
  if ( !v1 )
    goto LABEL_18;
  v10[0] = 0;
  v2 = 0;
  v3 = 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v1 + 112);
    if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) > 2u || (unsigned int)sub_736DD0(v1) )
      break;
    sub_7E4C20(*(_QWORD *)(v1 + 168), (__int64)a1, v10);
    v5 = *(_QWORD *)(v1 + 168);
    v10[0] = v1;
    v6 = *(_QWORD *)(v5 + 216);
    if ( !v6 )
      goto LABEL_4;
    if ( v3 )
      *(_QWORD *)(v2 + 112) = v6;
    else
      v3 = *(_QWORD *)(v5 + 216);
    do
    {
      v2 = v6;
      v6 = *(_QWORD *)(v6 + 112);
    }
    while ( v6 );
    *(_QWORD *)(v5 + 216) = 0;
    if ( !v4 )
      goto LABEL_15;
LABEL_5:
    v1 = v4;
  }
  v10[0] = v1;
LABEL_4:
  if ( v4 )
    goto LABEL_5;
  v1 = v10[0];
  if ( v3 )
  {
LABEL_15:
    *(_QWORD *)(v1 + 112) = v3;
    v4 = v3;
    v2 = 0;
    v3 = 0;
    goto LABEL_5;
  }
  v7 = sub_85EB10(a1);
  if ( v7 )
    *(_QWORD *)(v7 + 32) = v1;
LABEL_18:
  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_7E4E40(*(_QWORD *)(i + 128));
  }
  for ( j = (_QWORD *)a1[20]; j; j = (_QWORD *)*j )
    sub_7E4E40(j);
}
