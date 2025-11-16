// Function: sub_16C1FF0
// Address: 0x16c1ff0
//
__int64 sub_16C1FF0()
{
  __int64 v0; // rax
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 i; // rdi
  __int64 result; // rax
  __int64 v5; // r12
  int v6; // ebx
  int v7; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FA04D0, 1, 0) )
  {
    do
    {
      v6 = dword_4FA04D0;
      sub_16AF4B0();
      if ( v6 == 2 )
        break;
      v7 = dword_4FA04D0;
      sub_16AF4B0();
    }
    while ( v7 != 2 );
  }
  else
  {
    v0 = sub_22077B0(16);
    v1 = v0;
    if ( v0 )
    {
      sub_16C3010(v0, 1);
      *(_DWORD *)(v1 + 8) = 0;
      *(_BYTE *)(v1 + 12) = 1;
    }
    qword_4FA04D8 = v1;
    sub_16AF4B0();
    dword_4FA04D0 = 2;
  }
  v2 = qword_4FA04D8;
  sub_16C30C0(qword_4FA04D8);
  for ( i = qword_4FA04E0; qword_4FA04E0; i = qword_4FA04E0 )
    sub_16C1FB0(i);
  result = sub_16C30E0(v2);
  v5 = qword_4FA04D8;
  if ( qword_4FA04D8 )
  {
    sub_16C3090(qword_4FA04D8);
    result = j_j___libc_free_0(v5, 16);
    qword_4FA04D8 = 0;
  }
  return result;
}
