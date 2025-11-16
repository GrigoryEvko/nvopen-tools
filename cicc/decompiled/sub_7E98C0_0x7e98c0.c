// Function: sub_7E98C0
// Address: 0x7e98c0
//
void sub_7E98C0()
{
  _QWORD *v0; // rax
  __int64 v1; // rbx
  __int64 v2; // rbx
  __int64 v3; // rbx
  __int64 v4; // rbx
  __int64 i; // rbx

  v0 = (_QWORD *)unk_4F07288;
  v1 = *(_QWORD *)(unk_4F07288 + 96LL);
  if ( v1 )
  {
    do
    {
      sub_72B8C0(v1);
      *(_BYTE *)(v1 + 89) &= ~1u;
      v1 = *(_QWORD *)(v1 + 120);
    }
    while ( v1 );
    v0 = (_QWORD *)unk_4F07288;
  }
  v2 = v0[14];
  if ( v2 )
  {
    do
    {
      sub_72B8C0(v2);
      *(_BYTE *)(v2 + 89) &= ~1u;
      v2 = *(_QWORD *)(v2 + 112);
    }
    while ( v2 );
    v0 = (_QWORD *)unk_4F07288;
  }
  v3 = v0[18];
  if ( v3 )
  {
    do
    {
      sub_72B8C0(v3);
      *(_BYTE *)(v3 + 89) &= ~1u;
      v3 = *(_QWORD *)(v3 + 112);
    }
    while ( v3 );
    v0 = (_QWORD *)unk_4F07288;
  }
  v4 = v0[13];
  if ( v4 )
  {
    do
    {
      sub_72B8C0(v4);
      *(_BYTE *)(v4 + 89) &= ~1u;
      v4 = *(_QWORD *)(v4 + 112);
    }
    while ( v4 );
    v0 = (_QWORD *)unk_4F07288;
  }
  for ( i = v0[19]; i; i = *(_QWORD *)(i + 112) )
    sub_72B8C0(i);
}
