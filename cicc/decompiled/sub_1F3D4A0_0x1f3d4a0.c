// Function: sub_1F3D4A0
// Address: 0x1f3d4a0
//
__int64 __fastcall sub_1F3D4A0(_QWORD *a1, unsigned __int8 a2, __int64 a3, char a4)
{
  __int64 (*v4)(void); // rax
  unsigned __int8 v5; // al
  unsigned int v6; // r8d
  __int64 v8; // rdx
  _QWORD *v9; // r9
  unsigned int v10; // r8d

  v4 = *(__int64 (**)(void))(*a1 + 112LL);
  if ( (char *)v4 != (char *)sub_1F3CFB0 )
    return v4();
  v5 = a2;
  if ( !a2 || !a4 || *((_BYTE *)a1 + 259 * a2 + 2607) != 1 )
    return 1;
  v8 = a1[9258];
  if ( !v8 )
    goto LABEL_19;
  v9 = a1 + 9257;
  do
  {
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 <= 0xB8 || v10 == 185 && a2 > *(_BYTE *)(v8 + 36) )
    {
      v8 = *(_QWORD *)(v8 + 24);
    }
    else
    {
      v9 = (_QWORD *)v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
  }
  while ( v8 );
  if ( a1 + 9257 == v9 || *((_DWORD *)v9 + 8) > 0xB9u || *((_DWORD *)v9 + 8) == 185 && *((_BYTE *)v9 + 36) > a2 )
  {
LABEL_19:
    do
    {
      do
        ++v5;
      while ( !v5 );
    }
    while ( !a1[v5 + 15] || *((_BYTE *)a1 + 259 * v5 + 2607) == 1 );
  }
  else
  {
    v5 = *((_BYTE *)v9 + 40);
  }
  v6 = 0;
  if ( v5 != a4 )
    return 1;
  return v6;
}
