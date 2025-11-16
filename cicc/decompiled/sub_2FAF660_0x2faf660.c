// Function: sub_2FAF660
// Address: 0x2faf660
//
void __fastcall sub_2FAF660(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // r12
  unsigned int *v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r15
  char v13; // dl
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  char v17; // dl
  _QWORD *v18; // rax
  bool v19; // cf
  __int64 v20; // r14
  __int64 v21; // r14

  v6 = &a2[2 * a3];
  if ( a2 != v6 )
  {
    v7 = a2;
    while ( 1 )
    {
      v8 = *v7;
      v9 = *(_QWORD *)(a1[17] + 8 * v8);
      v10 = *v7;
      if ( !*((_BYTE *)v7 + 4) )
        goto LABEL_3;
      v11 = *(_QWORD *)(a1[1] + 8LL);
      v12 = *(unsigned int *)(v11 + 8LL * v10);
      sub_2FAF160((__int64)a1, *(_DWORD *)(v11 + 8LL * v10), v11, v8, a5, a6);
      v13 = *((_BYTE *)v7 + 4);
      v14 = (_QWORD *)(a1[3] + 112 * v12);
      if ( v13 == 2 )
        break;
      if ( v13 == 4 )
        goto LABEL_18;
      if ( v13 != 1 )
        goto LABEL_3;
      if ( __CFADD__(v14[1], v9) )
      {
        v14[1] = -1;
        goto LABEL_3;
      }
      v14[1] += v9;
      if ( !*((_BYTE *)v7 + 5) )
        goto LABEL_4;
LABEL_11:
      v15 = 2 * *v7 + 1;
      v16 = *(unsigned int *)(*(_QWORD *)(a1[1] + 8LL) + 4 * v15);
      sub_2FAF160((__int64)a1, *(_DWORD *)(*(_QWORD *)(a1[1] + 8LL) + 4 * v15), v15, v8, a5, a6);
      v17 = *((_BYTE *)v7 + 5);
      v18 = (_QWORD *)(a1[3] + 112 * v16);
      switch ( v17 )
      {
        case 2:
          v19 = __CFADD__(*v18, v9);
          v21 = *v18 + v9;
          if ( !v19 )
          {
            *v18 = v21;
            goto LABEL_4;
          }
LABEL_17:
          *v18 = -1;
          goto LABEL_4;
        case 4:
          goto LABEL_17;
        case 1:
          v19 = __CFADD__(v18[1], v9);
          v20 = v18[1] + v9;
          if ( v19 )
          {
            v18[1] = -1;
            goto LABEL_4;
          }
          v7 += 2;
          v18[1] = v20;
          if ( v6 == v7 )
            return;
          break;
        default:
LABEL_4:
          v7 += 2;
          if ( v6 == v7 )
            return;
          break;
      }
    }
    if ( __CFADD__(*v14, v9) )
LABEL_18:
      *v14 = -1;
    else
      *v14 += v9;
LABEL_3:
    if ( !*((_BYTE *)v7 + 5) )
      goto LABEL_4;
    goto LABEL_11;
  }
}
