// Function: sub_27DCB90
// Address: 0x27dcb90
//
__int64 __fastcall sub_27DCB90(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  unsigned int v10; // r8d
  int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 **v18; // rsi
  __int64 **v19; // rdx
  __int64 **v20; // rax
  __int64 v21; // rcx
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 *v24; // rax

  v4 = sub_AA54C0(a2);
  if ( !v4 )
    return 0;
  v5 = v4;
  v6 = v4 + 48;
  v7 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == v6 )
    goto LABEL_41;
  if ( !v7 )
    BUG();
  v8 = v7 - 24;
  v9 = *(unsigned __int8 *)(v7 - 24);
  if ( (unsigned int)(v9 - 30) > 0xA )
LABEL_41:
    BUG();
  if ( (unsigned int)(v9 - 29) <= 6 )
  {
    v10 = 0;
    if ( (unsigned int)(v9 - 29) > 4 )
      return v10;
LABEL_7:
    v11 = sub_B46E30(v8);
    if ( a2 == v5 )
      return 0;
    if ( v11 != 1 )
      return 0;
    if ( (*(_WORD *)(a2 + 2) & 0x7FFF) != 0 )
    {
      v14 = sub_ACC4F0(a2);
      sub_AD0030(v14);
      if ( *(_QWORD *)(v14 + 16) )
        return 0;
    }
    if ( *(_BYTE *)(a1 + 284) )
    {
      v16 = *(_QWORD **)(a1 + 264);
      v17 = &v16[*(unsigned int *)(a1 + 276)];
      if ( v16 != v17 )
      {
        while ( v5 != *v16 )
        {
          if ( v17 == ++v16 )
            goto LABEL_17;
        }
        return 0;
      }
    }
    else if ( sub_C8CA60(a1 + 256, v5) )
    {
      return 0;
    }
LABEL_17:
    if ( *(_BYTE *)(a1 + 124) )
    {
      v18 = *(__int64 ***)(a1 + 104);
      v19 = &v18[*(unsigned int *)(a1 + 116)];
      v20 = v18;
      if ( v18 == v19 )
        goto LABEL_28;
      while ( (__int64 *)v5 != *v20 )
      {
        if ( v19 == ++v20 )
          goto LABEL_28;
      }
      v21 = (unsigned int)(*(_DWORD *)(a1 + 116) - 1);
      *(_DWORD *)(a1 + 116) = v21;
      v22 = v18[v21];
      *v20 = v22;
      ++*(_QWORD *)(a1 + 96);
    }
    else
    {
      v24 = sub_C8CA60(a1 + 96, v5);
      if ( !v24 )
        goto LABEL_28;
      *v24 = -2;
      ++*(_DWORD *)(a1 + 120);
      ++*(_QWORD *)(a1 + 96);
    }
    if ( !*(_BYTE *)(a1 + 124) )
      goto LABEL_37;
    v23 = *(__int64 **)(a1 + 104);
    v21 = *(unsigned int *)(a1 + 116);
    v22 = &v23[v21];
    if ( v23 != v22 )
    {
      while ( a2 != *v23 )
      {
        if ( v22 == ++v23 )
          goto LABEL_38;
      }
      goto LABEL_28;
    }
LABEL_38:
    if ( (unsigned int)v21 < *(_DWORD *)(a1 + 112) )
    {
      *(_DWORD *)(a1 + 116) = v21 + 1;
      *v22 = a2;
      ++*(_QWORD *)(a1 + 96);
    }
    else
    {
LABEL_37:
      sub_C8CC70(a1 + 96, a2, (__int64)v22, v21, v12, v13);
    }
LABEL_28:
    sub_22C2BE0(*(_QWORD *)(a1 + 32), v5);
    sub_F50FD0(a2, *(_QWORD *)(a1 + 48));
    v10 = sub_98CE00(a2);
    if ( !(_BYTE)v10 )
    {
      sub_22C2BE0(*(_QWORD *)(a1 + 32), a2);
      return 1;
    }
    return v10;
  }
  if ( (unsigned int)(v9 - 37) > 3 )
    goto LABEL_7;
  return 0;
}
