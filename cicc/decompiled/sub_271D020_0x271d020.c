// Function: sub_271D020
// Address: 0x271d020
//
__int64 __fastcall sub_271D020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r14
  __int64 v11; // rsi
  __int64 *v12; // r13
  int v13; // edx
  __int64 *v14; // rax
  __int64 *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 *v19; // r13
  int v21; // edi
  _QWORD *v22; // rax
  __int64 *v23; // rax
  char v24; // di
  _QWORD *v25; // rax
  __int64 *v26; // rax

  v6 = a2;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
    *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)a1 &= *(_BYTE *)a2;
  *(_BYTE *)(a1 + 1) &= *(_BYTE *)(a2 + 1);
  *(_BYTE *)(a1 + 112) |= *(_BYTE *)(a2 + 112);
  v8 = *(__int64 **)(a2 + 24);
  if ( *(_BYTE *)(a2 + 44) )
    v9 = *(unsigned int *)(a2 + 36);
  else
    v9 = *(unsigned int *)(a2 + 32);
  v10 = &v8[v9];
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      v11 = *v8;
      v12 = v8;
      if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v8 )
        goto LABEL_8;
    }
    if ( v8 != v10 )
    {
      v24 = *(_BYTE *)(a1 + 44);
      if ( !v24 )
        goto LABEL_41;
LABEL_31:
      v25 = *(_QWORD **)(a1 + 24);
      a4 = *(unsigned int *)(a1 + 36);
      v9 = (__int64)&v25[a4];
      if ( v25 == (_QWORD *)v9 )
      {
LABEL_42:
        if ( (unsigned int)a4 < *(_DWORD *)(a1 + 32) )
        {
          a4 = (unsigned int)(a4 + 1);
          *(_DWORD *)(a1 + 36) = a4;
          *(_QWORD *)v9 = v11;
          v24 = *(_BYTE *)(a1 + 44);
          ++*(_QWORD *)(a1 + 16);
          goto LABEL_35;
        }
        goto LABEL_41;
      }
      while ( *v25 != v11 )
      {
        if ( (_QWORD *)v9 == ++v25 )
          goto LABEL_42;
      }
LABEL_35:
      while ( 1 )
      {
        v26 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        v11 = *v26;
        for ( ++v12; (unsigned __int64)*v26 >= 0xFFFFFFFFFFFFFFFELL; v12 = v26 )
        {
          if ( v10 == ++v26 )
            goto LABEL_8;
          v11 = *v26;
        }
        if ( v12 == v10 )
          break;
        if ( v24 )
          goto LABEL_31;
LABEL_41:
        sub_C8CC70(a1 + 16, v11, v9, a4, a5, a6);
        v24 = *(_BYTE *)(a1 + 44);
      }
    }
  }
LABEL_8:
  v13 = *(_DWORD *)(v6 + 84);
  v14 = *(__int64 **)(v6 + 72);
  if ( *(_BYTE *)(v6 + 92) )
    v15 = &v14[v13];
  else
    v15 = &v14[*(unsigned int *)(v6 + 80)];
  v16 = (unsigned int)(v13 - *(_DWORD *)(v6 + 88));
  v17 = (unsigned int)(*(_DWORD *)(a1 + 84) - *(_DWORD *)(a1 + 88));
  LOBYTE(v6) = (_DWORD)v17 != (_DWORD)v16;
  if ( v14 != v15 )
  {
    while ( 1 )
    {
      v18 = *v14;
      v19 = v14;
      if ( (unsigned __int64)*v14 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v15 == ++v14 )
        return (unsigned int)v6;
    }
    if ( v15 != v14 )
    {
      v21 = *(unsigned __int8 *)(a1 + 92);
      if ( !(_BYTE)v21 )
        goto LABEL_26;
LABEL_16:
      v22 = *(_QWORD **)(a1 + 72);
      v17 = *(unsigned int *)(a1 + 84);
      v16 = (__int64)&v22[v17];
      if ( v22 == (_QWORD *)v16 )
      {
LABEL_27:
        if ( (unsigned int)v17 < *(_DWORD *)(a1 + 80) )
        {
          v17 = (unsigned int)(v17 + 1);
          LODWORD(v6) = v21;
          *(_DWORD *)(a1 + 84) = v17;
          *(_QWORD *)v16 = v18;
          v21 = *(unsigned __int8 *)(a1 + 92);
          ++*(_QWORD *)(a1 + 64);
          goto LABEL_20;
        }
        goto LABEL_26;
      }
      while ( *v22 != v18 )
      {
        if ( (_QWORD *)v16 == ++v22 )
          goto LABEL_27;
      }
LABEL_20:
      while ( 1 )
      {
        v23 = v19 + 1;
        if ( v19 + 1 == v15 )
          break;
        v18 = *v23;
        for ( ++v19; (unsigned __int64)*v23 >= 0xFFFFFFFFFFFFFFFELL; v19 = v23 )
        {
          if ( v15 == ++v23 )
            return (unsigned int)v6;
          v18 = *v23;
        }
        if ( v15 == v19 )
          return (unsigned int)v6;
        if ( (_BYTE)v21 )
          goto LABEL_16;
LABEL_26:
        sub_C8CC70(a1 + 64, v18, v16, v17, a5, a6);
        v21 = *(unsigned __int8 *)(a1 + 92);
        LODWORD(v6) = v16 | v6;
      }
    }
  }
  return (unsigned int)v6;
}
