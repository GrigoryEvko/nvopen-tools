// Function: sub_BD2CC0
// Address: 0xbd2cc0
//
_QWORD *__fastcall sub_BD2CC0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  char v8; // cl

  v2 = 32LL * (unsigned int)a2 + a1;
  v3 = 4LL * (unsigned int)a2;
  if ( HIDWORD(a2) )
  {
    v4 = (unsigned int)(HIDWORD(a2) + 8);
    v5 = sub_22077B0(v4 + v2);
    v6 = (_QWORD *)(v4 + v5);
    v7 = (_QWORD *)(v4 + v5 + v3 * 8);
    *((_DWORD *)v7 + 1) = *((_DWORD *)v7 + 1) & 0x38000000 | a2 & 0x7FFFFFF | 0x80000000;
    if ( (_QWORD *)(v4 + v5) == v7 )
    {
LABEL_10:
      *(_QWORD *)(v5 + HIDWORD(a2)) = HIDWORD(a2);
      return v7;
    }
    v8 = 1;
  }
  else
  {
    v6 = (_QWORD *)sub_22077B0(v2);
    v7 = &v6[v3];
    *((_DWORD *)v7 + 1) = *((_DWORD *)v7 + 1) & 0x38000000 | a2 & 0x7FFFFFF;
    if ( v6 == v7 )
      return v7;
    v5 = (__int64)v6;
    v8 = 0;
  }
  do
  {
    if ( v6 )
    {
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      v6[3] = v7;
    }
    v6 += 4;
  }
  while ( v6 != v7 );
  if ( v8 )
    goto LABEL_10;
  return v7;
}
