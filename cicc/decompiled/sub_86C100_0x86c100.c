// Function: sub_86C100
// Address: 0x86c100
//
void __fastcall sub_86C100(__int64 a1)
{
  char v1; // al
  __int64 v2; // r12
  __int64 v3; // r15
  _BYTE *v4; // rax
  __int64 v5; // r13
  _BOOL4 v6; // [rsp-3Ch] [rbp-3Ch]

  v1 = *(_BYTE *)(a1 + 72);
  if ( (v1 & 4) != 0 )
    return;
  v6 = 1;
  v2 = *(_QWORD *)(a1 + 56);
  if ( *(_QWORD *)(a1 + 16) && !*(_QWORD *)(v2 + 24) )
  {
    v3 = qword_4F06BC0;
    v6 = (v1 & 0x20) != 0;
    if ( v2 == qword_4F06BC0 )
      goto LABEL_5;
    do
    {
LABEL_13:
      v5 = *(_QWORD *)(v3 + 32);
      if ( (unsigned int)sub_733F40() )
      {
        v6 = 1;
      }
      else if ( *(_QWORD *)(a1 + 64) || (*(_BYTE *)(a1 + 72) & 1) != 0 )
      {
        sub_86C0B0(*(__int64 **)a1, v3, v5);
      }
      v3 = qword_4F06BC0;
    }
    while ( qword_4F06BC0 != v2 );
    goto LABEL_5;
  }
  v3 = qword_4F06BC0;
  if ( v2 != qword_4F06BC0 )
    goto LABEL_13;
LABEL_5:
  if ( *(_QWORD *)(a1 + 64) || (*(_BYTE *)(a1 + 72) & 1) != 0 )
  {
    v4 = *(_BYTE **)(v2 + 32);
    *(_BYTE *)(v2 + 1) |= 4u;
    if ( *v4 != 4 && !v6 )
    {
      if ( (unsigned int)sub_733920(v2) )
      {
        sub_86C0B0(*(__int64 **)a1, v2, *(_QWORD *)(v2 + 32));
        *(_QWORD *)(a1 + 56) = 0;
      }
    }
  }
}
