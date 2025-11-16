// Function: sub_1D50960
// Address: 0x1d50960
//
void __fastcall sub_1D50960(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // rdi
  __int64 v7; // r13
  _QWORD *v8; // rdx
  __int64 v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // r12
  int v15; // edx
  __int64 v16; // rbx
  int v17; // r15d
  __int64 v18; // rdx

  *(_BYTE *)(*(_QWORD *)(a1 + 272) + 658LL) = 0;
  if ( a3 != a2 )
  {
    v5 = *(_QWORD *)(a1 + 280);
    v7 = a2;
    do
    {
      if ( *(_BYTE *)(v5 + 760) )
        break;
      v8 = *(_QWORD **)(a1 + 352);
      a2 = v7 - 24;
      v9 = 0;
      v10 = *(_QWORD **)(a1 + 344);
      if ( v7 )
        v9 = v7 - 24;
      if ( v8 == v10 )
      {
        v11 = &v10[*(unsigned int *)(a1 + 364)];
        if ( v10 == v11 )
        {
          v13 = *(_QWORD **)(a1 + 344);
        }
        else
        {
          do
          {
            if ( v9 == *v10 )
              break;
            ++v10;
          }
          while ( v11 != v10 );
          v13 = v11;
        }
        goto LABEL_13;
      }
      a2 = v9;
      v11 = &v8[*(unsigned int *)(a1 + 360)];
      v10 = sub_16CC9F0(a1 + 336, v9);
      if ( v9 == *v10 )
      {
        v18 = *(_QWORD *)(a1 + 352);
        if ( v18 != *(_QWORD *)(a1 + 344) )
        {
          a2 = *(unsigned int *)(a1 + 360);
          v13 = (_QWORD *)(v18 + 8 * a2);
          goto LABEL_10;
        }
        a2 = *(unsigned int *)(a1 + 364);
        v5 = *(_QWORD *)(a1 + 280);
        v13 = (_QWORD *)(v18 + 8 * a2);
      }
      else
      {
        v12 = *(_QWORD *)(a1 + 352);
        if ( v12 != *(_QWORD *)(a1 + 344) )
        {
          v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a1 + 360));
          v13 = v10;
LABEL_10:
          v5 = *(_QWORD *)(a1 + 280);
          goto LABEL_13;
        }
        v5 = *(_QWORD *)(a1 + 280);
        v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a1 + 364));
        v13 = v10;
      }
LABEL_13:
      while ( v13 != v10 && *v10 >= 0xFFFFFFFFFFFFFFFELL )
        ++v10;
      if ( v10 == v11 )
      {
        a2 = v9;
        sub_2093EA0(v5, v9);
        v5 = *(_QWORD *)(a1 + 280);
      }
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( a3 != v7 );
  }
  v14 = *(_QWORD *)(a1 + 272);
  v16 = sub_2051DF0();
  v17 = v15;
  if ( v16 )
  {
    nullsub_686();
    a2 = 0;
    *(_QWORD *)(v14 + 176) = v16;
    *(_DWORD *)(v14 + 184) = v17;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v14 + 176) = 0;
    *(_DWORD *)(v14 + 184) = v15;
  }
  *a4 = *(_BYTE *)(*(_QWORD *)(a1 + 280) + 760LL);
  sub_20515F0(*(_QWORD *)(a1 + 280), a2);
  sub_1D50350(a1);
}
