// Function: sub_137D930
// Address: 0x137d930
//
_QWORD *__fastcall sub_137D930(__int64 a1, __int64 a2)
{
  int v2; // eax
  _QWORD *v3; // r8
  int v4; // ecx
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  _QWORD *v9; // rax
  int v11; // eax
  int v12; // r9d

  v2 = *(_DWORD *)(a1 + 24);
  v3 = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(a1 + 8);
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v9 = (_QWORD *)v7[1];
      if ( v9 )
      {
        do
        {
          v3 = v9;
          v9 = (_QWORD *)*v9;
        }
        while ( v9 );
        return v3;
      }
    }
    else
    {
      v11 = 1;
      while ( v8 != -8 )
      {
        v12 = v11 + 1;
        v6 = v4 & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v11 = v12;
      }
    }
    return 0;
  }
  return v3;
}
