// Function: sub_BDB7C0
// Address: 0xbdb7c0
//
__int64 __fastcall sub_BDB7C0(_BYTE *a1, __int64 a2)
{
  _BYTE *v2; // rbx
  bool v3; // al
  int v4; // eax
  _BYTE *v5; // r13
  __int64 v6; // rbx
  unsigned int v7; // r8d
  unsigned int v9; // edx
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // r14d

  v2 = a1;
  v3 = (*(a1 - 16) & 2) != 0;
  while ( 1 )
  {
    if ( v3 )
    {
      v4 = *((_DWORD *)v2 - 6);
      if ( (unsigned int)(v4 - 2) > 1 )
        return 0;
      v5 = v2 - 16;
      v6 = *((_QWORD *)v2 - 4);
    }
    else
    {
      v4 = (*((_WORD *)v2 - 8) >> 6) & 0xF;
      if ( (unsigned int)(v4 - 2) > 1 )
        return 0;
      v5 = v2 - 16;
      v6 = (__int64)&v2[-8 * ((*(v2 - 16) >> 2) & 0xF) - 16];
    }
    if ( **(_BYTE **)v6 )
      return 0;
    if ( v4 != 3 )
      goto LABEL_7;
    v11 = *(_QWORD *)(v6 + 16);
    if ( *(_BYTE *)v11 != 1 )
      return 0;
    v12 = *(_QWORD *)(v11 + 136);
    if ( *(_BYTE *)v12 != 17 )
      return 0;
    v13 = *(_DWORD *)(v12 + 32);
    if ( v13 <= 0x40 )
    {
      if ( *(_QWORD *)(v12 + 24) )
        return 0;
    }
    else if ( v13 != (unsigned int)sub_C444A0(v12 + 24) )
    {
      return 0;
    }
    if ( **(_BYTE **)sub_A17150(v5) )
      return 0;
LABEL_7:
    v2 = *(_BYTE **)(v6 + 8);
    if ( !v2 )
      return 0;
    if ( (unsigned __int8)(*v2 - 5) > 0x1Fu )
      return 0;
    sub_AE6EC0(a2, (__int64)v2);
    v7 = v9;
    if ( !(_BYTE)v9 )
      return 0;
    v3 = (*(v2 - 16) & 2) != 0;
    if ( (*(v2 - 16) & 2) != 0 )
      v10 = *((_DWORD *)v2 - 6);
    else
      v10 = (*((_WORD *)v2 - 8) >> 6) & 0xF;
    if ( v10 <= 1 )
      return v7;
  }
}
