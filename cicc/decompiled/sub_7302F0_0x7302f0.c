// Function: sub_7302F0
// Address: 0x7302f0
//
void sub_7302F0()
{
  size_t v0; // rdi
  __int64 *v1; // rsi
  __int64 v2; // rdx
  __int64 *v3; // rcx
  __int64 v4; // rax
  __int64 i; // rdx
  __int64 v6; // rax

  if ( !qword_4F07A48 )
    return;
  qsort(qword_4F07A68, qword_4F07A48, 0x18u, (__compar_fn_t)sub_728140);
  v0 = qword_4F07A48;
  v1 = (__int64 *)((char *)qword_4F07A68 + 24 * qword_4F07A48 - 24);
  v2 = *v1;
LABEL_3:
  v3 = (__int64 *)(qword_4F07A60 + 144);
  v4 = *(_QWORD *)(qword_4F07A60 + 144);
  while ( v2 != v4 )
  {
    v3 = (__int64 *)(v4 + 112);
LABEL_5:
    v4 = *v3;
    if ( !*v3 )
      goto LABEL_3;
  }
  *v3 = *(_QWORD *)(v2 + 112);
  *(_QWORD *)(v2 + 112) = *(_QWORD *)(v1[1] + 112);
  *(_QWORD *)(v1[1] + 112) = v2;
  if ( *(_QWORD *)(v2 + 112) )
  {
    if ( !--v0 )
      goto LABEL_11;
    goto LABEL_9;
  }
  *(_QWORD *)(qword_4F07A58 + 48) = v2;
  if ( --v0 )
  {
LABEL_9:
    v2 = *(v1 - 3);
    v1 -= 3;
    goto LABEL_5;
  }
LABEL_11:
  for ( i = qword_4F07A60 + 144; qword_4F07A48; --qword_4F07A48 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)i;
      if ( *(char **)(*(_QWORD *)i + 8LL) == off_4B6D4E0 )
        break;
      i = v6 + 112;
      if ( !qword_4F07A48 )
        goto LABEL_16;
    }
    *(_QWORD *)i = *(_QWORD *)(v6 + 112);
  }
LABEL_16:
  qword_4F07A60 = 0;
  qword_4F07A58 = 0;
}
