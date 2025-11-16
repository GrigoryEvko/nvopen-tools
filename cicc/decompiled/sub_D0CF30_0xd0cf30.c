// Function: sub_D0CF30
// Address: 0xd0cf30
//
_QWORD *__fastcall sub_D0CF30(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // r8
  _QWORD *v9; // rax
  int v11; // eax
  int v12; // r9d

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = v2 - 1;
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v11 = 1;
    while ( v7 != -4096 )
    {
      v12 = v11 + 1;
      v5 = v4 & (v11 + v5);
      v6 = (__int64 *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_3;
      v11 = v12;
    }
    return 0;
  }
LABEL_3:
  v8 = (_QWORD *)v6[1];
  if ( v8 )
  {
    v9 = (_QWORD *)v6[1];
    do
    {
      v8 = v9;
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
  }
  return v8;
}
