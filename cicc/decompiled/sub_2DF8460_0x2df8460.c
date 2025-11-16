// Function: sub_2DF8460
// Address: 0x2df8460
//
__int64 __fastcall sub_2DF8460(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r8
  _QWORD *v3; // rcx
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r10
  unsigned int v10; // r9d
  _QWORD *v11; // rdx
  _QWORD *v12; // rbx
  int v14; // edx
  int v15; // r12d

  v2 = a2[3];
  v3 = *(_QWORD **)(v2 + 56);
  if ( a2 == v3 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v2 + 24));
  while ( 1 )
  {
    v4 = (_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
    v5 = v4;
    if ( !v4 )
      BUG();
    a2 = (_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
    v6 = *v4;
    if ( (v6 & 4) == 0 && (*((_BYTE *)v5 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
        a2 = (_QWORD *)v7;
        if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
          break;
        v6 = *(_QWORD *)v7;
      }
    }
    v8 = *(unsigned int *)(a1 + 144);
    v9 = *(_QWORD *)(a1 + 128);
    if ( (_DWORD)v8 )
    {
      v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = (_QWORD *)*v11;
      if ( a2 != (_QWORD *)*v11 )
      {
        v14 = 1;
        while ( v12 != (_QWORD *)-4096LL )
        {
          v15 = v14 + 1;
          v10 = (v8 - 1) & (v14 + v10);
          v11 = (_QWORD *)(v9 + 16LL * v10);
          v12 = (_QWORD *)*v11;
          if ( a2 == (_QWORD *)*v11 )
            goto LABEL_10;
          v14 = v15;
        }
        goto LABEL_11;
      }
LABEL_10:
      if ( v11 != (_QWORD *)(v9 + 16 * v8) )
        return v11[1];
    }
LABEL_11:
    if ( v3 == a2 )
      return *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v2 + 24));
  }
}
