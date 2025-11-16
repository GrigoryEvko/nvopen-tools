// Function: sub_738240
// Address: 0x738240
//
void sub_738240()
{
  __int64 v0; // rbx
  _QWORD *v1; // rbx
  _QWORD *v2; // rdi
  unsigned __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rdx
  void *v6; // rax
  void *v7; // rax

  v0 = unk_4F07310;
  if ( unk_4F07310 && unk_4F07310 > (unsigned __int64)qword_4F07AF8 )
  {
    v7 = (void *)sub_822C60(base, 8 * qword_4F07AF8, 8LL * unk_4F07310);
    qword_4F07AF8 = v0;
    base = v7;
  }
  v1 = (_QWORD *)unk_4F07308;
  if ( unk_4F07308 )
  {
    v2 = base;
    v3 = 0;
    do
    {
      if ( qword_4F07AF8 <= v3 )
      {
        if ( qword_4F07AF8 )
        {
          v4 = 2 * qword_4F07AF8;
          v5 = 16 * qword_4F07AF8;
        }
        else
        {
          v5 = 0x2000;
          v4 = 1024;
        }
        v6 = (void *)sub_822C60(v2, 8 * qword_4F07AF8, v5);
        qword_4F07AF8 = v4;
        base = v6;
        v2 = v6;
      }
      v2[v3++] = v1;
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
  else
  {
    v3 = 0;
  }
  unk_4F07310 = v3;
  dword_4F07B00 = 1;
  qword_4F07AE8 = 0;
}
