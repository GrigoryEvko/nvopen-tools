// Function: sub_255DB00
// Address: 0x255db00
//
__int64 __fastcall sub_255DB00(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v3; // r11d
  int v6; // edx
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // esi
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r10
  _QWORD *v15; // r14
  int v16; // ebx
  int v17; // ecx
  __int64 v19; // [rsp-8h] [rbp-8h]

  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    *a3 = 0;
    return 0;
  }
  *((_DWORD *)&v19 - 11) = 1;
  v7 = *a2;
  v8 = a2[1];
  v9 = v6 - 1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = (v6 - 1)
      & ((unsigned int)((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v8) | ((unsigned __int64)(unsigned int)(37 * v7) << 32))) >> 31)
       ^ (756364221 * (_DWORD)v8));
  v12 = (_QWORD *)(v10 + 96 * v11);
  v13 = v12[1];
  v14 = *v12;
  v15 = 0;
  LOBYTE(v3) = v8 == v13 && v7 == *v12;
  if ( (_BYTE)v3 )
  {
LABEL_11:
    *a3 = v12;
    return 1;
  }
  while ( v14 != 0x7FFFFFFFFFFFFFFFLL )
  {
    if ( v14 == 0x7FFFFFFFFFFFFFFELL && v13 == 0x7FFFFFFFFFFFFFFELL && !v15 )
      v15 = v12;
LABEL_7:
    v16 = *((_DWORD *)&v19 - 11);
    v17 = v16 + 1;
    LODWORD(v11) = v9 & (v16 + v11);
    v12 = (_QWORD *)(v10 + 96LL * (unsigned int)v11);
    v13 = v12[1];
    v14 = *v12;
    if ( v13 == v8 && v7 == v14 )
      goto LABEL_11;
    *((_DWORD *)&v19 - 11) = v17;
  }
  if ( v13 != 0x7FFFFFFFFFFFFFFFLL )
    goto LABEL_7;
  if ( !v15 )
    v15 = v12;
  *a3 = v15;
  return v3;
}
