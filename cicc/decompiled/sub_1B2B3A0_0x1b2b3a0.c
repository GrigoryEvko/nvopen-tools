// Function: sub_1B2B3A0
// Address: 0x1b2b3a0
//
__int64 __fastcall sub_1B2B3A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // r8d
  __int64 v6; // r13
  _QWORD *v7; // rax
  _QWORD *v9; // rcx
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13[7]; // [rsp-38h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v3 )
    return 0;
  v4 = *(_QWORD *)a2 + 48 * v3 - 48;
  v5 = *(unsigned __int8 *)(v4 + 40);
  if ( (_BYTE)v5 )
  {
    v6 = *(_QWORD *)(a3 + 24);
    if ( v6 )
    {
      v7 = sub_1648700(*(_QWORD *)(a3 + 24));
      v5 = 0;
      if ( *((_BYTE *)v7 + 16) == 77 )
      {
        if ( (*((_BYTE *)v7 + 23) & 0x40) != 0 )
          v9 = (_QWORD *)*(v7 - 1);
        else
          v9 = &v7[-3 * (*((_DWORD *)v7 + 5) & 0xFFFFFFF)];
        v10 = *(_QWORD *)(v4 + 32);
        v5 = 0;
        if ( *(_QWORD *)(v10 + 48) == v9[3 * *((unsigned int *)v7 + 14)
                                       + 1
                                       + -1431655765 * (unsigned int)((v6 - (__int64)v9) >> 3)] )
        {
          v11 = *(_QWORD *)(a1 + 24);
          v12 = *(_QWORD *)(v10 + 56);
          v13[0] = *(_QWORD *)(v10 + 48);
          v13[1] = v12;
          return (unsigned int)sub_15CCFD0(v11, v13, v6);
        }
      }
    }
    else
    {
      return 0;
    }
    return v5;
  }
  if ( *(_DWORD *)a3 < *(_DWORD *)v4 )
    return v5;
  LOBYTE(v5) = *(_DWORD *)(a3 + 4) <= *(_DWORD *)(v4 + 4);
  return v5;
}
