// Function: sub_27306B0
// Address: 0x27306b0
//
unsigned __int8 *__fastcall sub_27306B0(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  unsigned __int8 *v3; // rcx
  _BYTE *v4; // rax
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rbx
  __int64 *i; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rdx

  if ( a3 == -1 )
  {
    v6 = *a2;
    if ( (_BYTE)v6 == 84 )
    {
LABEL_12:
      v9 = *((_QWORD *)a2 + 5);
      goto LABEL_13;
    }
  }
  else
  {
    if ( (a2[7] & 0x40) != 0 )
      v3 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v3 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v4 = *(_BYTE **)&v3[32 * a3];
    if ( (unsigned __int8)(*v4 - 67) <= 0xCu )
      return v4 + 24;
    v6 = *a2;
    if ( (_BYTE)v6 == 84 )
    {
      v9 = *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + 8LL * a3);
      v19 = sub_AA4FF0(v9);
      if ( !v19 )
        BUG();
      v20 = (unsigned int)*(unsigned __int8 *)(v19 - 24) - 39;
      if ( (unsigned int)v20 > 0x38 || (v21 = 0x100060000000001LL, !_bittest64(&v21, v20)) )
      {
        v17 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v17 != v9 + 48 )
        {
          if ( !v17 )
            BUG();
          goto LABEL_22;
        }
LABEL_26:
        v4 = 0;
        return v4 + 24;
      }
LABEL_13:
      v10 = *(_QWORD *)(a1 + 8);
      if ( v9 )
      {
        v11 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
        v12 = *(_DWORD *)(v9 + 44) + 1;
      }
      else
      {
        v11 = 0;
        v12 = 0;
      }
      if ( v12 >= *(_DWORD *)(v10 + 32) )
        BUG();
      v13 = 0x100060000000001LL;
      for ( i = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v10 + 24) + 8 * v11) + 8LL); ; i = (__int64 *)i[1] )
      {
        v15 = sub_AA4FF0(*i);
        if ( !v15 )
          BUG();
        v16 = (unsigned int)*(unsigned __int8 *)(v15 - 24) - 39;
        if ( (unsigned int)v16 > 0x38 || !_bittest64(&v13, v16) )
          break;
      }
      v17 = *(_QWORD *)(*i + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17 != *i + 48 )
      {
        if ( !v17 )
          BUG();
LABEL_22:
        v18 = *(unsigned __int8 *)(v17 - 24);
        v4 = (_BYTE *)(v17 - 24);
        if ( (unsigned int)(v18 - 30) >= 0xB )
          v4 = 0;
        return v4 + 24;
      }
      goto LABEL_26;
    }
  }
  v7 = (unsigned int)(v6 - 39);
  if ( (unsigned int)v7 <= 0x38 )
  {
    v8 = 0x100060000000001LL;
    if ( _bittest64(&v8, v7) )
      goto LABEL_12;
  }
  return a2 + 24;
}
