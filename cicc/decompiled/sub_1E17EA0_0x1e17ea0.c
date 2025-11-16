// Function: sub_1E17EA0
// Address: 0x1e17ea0
//
__int64 __fastcall sub_1E17EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdx
  int v12; // eax
  int v13; // r12d

  v6 = 0;
  v7 = *(unsigned int *)(a1 + 40);
  v8 = *(_QWORD *)(a1 + 16);
  if ( !(_DWORD)v7 )
    return 0;
  while ( 1 )
  {
    v9 = *(_QWORD *)(a1 + 32) + 40 * v6;
    if ( *(_BYTE *)v9 || (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
      goto LABEL_8;
    v10 = *(unsigned __int16 *)(v9 + 2);
    LOWORD(v10) = v10 & 0xFF0;
    v11 = v10;
    if ( *(unsigned __int16 *)(v8 + 2) > (unsigned int)v6
      && (v12 = *(_DWORD *)(*(_QWORD *)(v8 + 40) + 8 * v6 + 4), (v12 & 1) != 0) )
    {
      v13 = BYTE2(v12);
      if ( !(_WORD)v11 )
        return 1;
    }
    else
    {
      if ( !(_WORD)v11 )
        goto LABEL_8;
      v13 = -1;
    }
    if ( (unsigned int)sub_1E16AB0(a1, v6, v11, a4, a5, a6) != v13 )
      return 1;
LABEL_8:
    if ( v7 == ++v6 )
      return 0;
  }
}
