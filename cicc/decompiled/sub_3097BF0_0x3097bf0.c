// Function: sub_3097BF0
// Address: 0x3097bf0
//
void __fastcall sub_3097BF0(__int64 a1, _DWORD *a2, _QWORD *a3, __int64 (__fastcall ***a4)(_QWORD, __int64))
{
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r14
  _QWORD *v9; // r10
  int v10; // r13d
  _BYTE *v11; // rsi
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  char v14; // al
  __int64 v15; // rdx
  unsigned __int64 v16; // r9
  __int64 v17; // rax
  _BYTE *v18; // r9
  _BYTE *v19; // rdi
  __int64 v20; // rax
  _QWORD *v22; // [rsp+18h] [rbp-58h]
  _QWORD *v23; // [rsp+18h] [rbp-58h]
  _QWORD *v24; // [rsp+18h] [rbp-58h]
  _BYTE *v25; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v26; // [rsp+28h] [rbp-48h]
  _BYTE *v27; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD **)(a1 + 192);
  v7 = &v6[*(unsigned int *)(a1 + 200)];
  v25 = 0;
  v26 = 0;
  v27 = 0;
  if ( v6 == v7 )
  {
    *a2 = 0;
  }
  else
  {
    while ( 1 )
    {
      v8 = *v6;
      v9 = v6;
      if ( *v6 )
        break;
      if ( v7 == ++v6 )
        goto LABEL_29;
    }
    v10 = 0;
    if ( v7 == v6 )
    {
LABEL_29:
      *a2 = 0;
      return;
    }
    while ( 1 )
    {
      if ( !a4 || (v22 = v9, v14 = (**a4)(a4, v8), v9 = v22, !v14) )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v8 + 40LL) - 4) <= 1 )
          break;
        v23 = v9;
        v20 = sub_22F59B0(*(_QWORD *)(v8 + 8), *(unsigned __int16 *)(*(_QWORD *)v8 + 56LL));
        v9 = v23;
        if ( *(_DWORD *)(a1 + 176) != *(_DWORD *)(v20 + 40) )
          break;
      }
      v11 = v26;
      v12 = (_QWORD *)(*a3 + 8LL * *(unsigned int *)(v8 + 40));
      if ( v26 == v27 )
      {
        v24 = v9;
        sub_12ECAA0((__int64)&v25, v26, v12);
        v9 = v24;
      }
      else
      {
        if ( v26 )
        {
          *(_QWORD *)v26 = *v12;
          v11 = v26;
        }
        v26 = v11 + 8;
      }
      v13 = v9 + 1;
      if ( v7 == v9 + 1 )
        goto LABEL_16;
LABEL_10:
      while ( 1 )
      {
        v8 = *v13;
        v9 = v13;
        if ( *v13 )
          break;
        if ( v7 == ++v13 )
          goto LABEL_16;
      }
      if ( v7 == v13 )
        goto LABEL_16;
    }
    v15 = v10++;
    *(_QWORD *)(*a3 + 8 * v15) = *(_QWORD *)(*a3 + 8LL * *(unsigned int *)(v8 + 40));
    v13 = v9 + 1;
    if ( v7 != v9 + 1 )
      goto LABEL_10;
LABEL_16:
    v16 = (unsigned __int64)v25;
    if ( v26 != v25 )
    {
      v17 = 8LL * v10;
      v18 = &v25[-v17];
      v19 = &v26[v17 - (_QWORD)v25];
      do
      {
        *(_QWORD *)(*a3 + v17) = *(_QWORD *)&v18[v17];
        v17 += 8;
      }
      while ( v19 != (_BYTE *)v17 );
      v16 = (unsigned __int64)v25;
    }
    *a2 = v10;
    if ( v16 )
      j_j___libc_free_0(v16);
  }
}
