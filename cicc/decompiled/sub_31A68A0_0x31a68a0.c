// Function: sub_31A68A0
// Address: 0x31a68a0
//
__int64 __fastcall sub_31A68A0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdx
  _QWORD *v5; // rdi
  unsigned int v6; // r8d
  unsigned int v7; // edx
  _QWORD *v8; // rax
  _BYTE *v9; // r9
  int v10; // eax
  int v11; // r10d

  if ( !a2 )
    return 0;
  if ( *a2 != 84 )
    return 0;
  v3 = *(_QWORD *)(a1 + 136);
  v4 = *(unsigned int *)(a1 + 152);
  v5 = (_QWORD *)(v3 + 16 * v4);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v3 + 16LL * v7);
  v9 = (_BYTE *)*v8;
  if ( a2 != (_BYTE *)*v8 )
  {
    v10 = 1;
    while ( v9 != (_BYTE *)-4096LL )
    {
      v11 = v10 + 1;
      v7 = v6 & (v10 + v7);
      v8 = (_QWORD *)(v3 + 16LL * v7);
      v9 = (_BYTE *)*v8;
      if ( a2 == (_BYTE *)*v8 )
        goto LABEL_6;
      v10 = v11;
    }
    return 0;
  }
LABEL_6:
  LOBYTE(v6) = v5 != v8;
  return v6;
}
