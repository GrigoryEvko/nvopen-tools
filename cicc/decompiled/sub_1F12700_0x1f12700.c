// Function: sub_1F12700
// Address: 0x1f12700
//
void __fastcall sub_1F12700(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int *v6; // r13
  unsigned int *v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r12
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r15
  char v14; // al
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r15
  char v18; // al
  _QWORD *v19; // rdi

  v6 = &a2[2 * a3];
  if ( v6 != a2 )
  {
    v8 = a2;
    while ( 1 )
    {
      v9 = *v8;
      v10 = *(_QWORD *)(a1[47] + 8 * v9);
      v11 = *v8;
      if ( !*((_BYTE *)v8 + 4) )
        goto LABEL_3;
      v12 = *(_QWORD *)(a1[30] + 240LL);
      v13 = *(unsigned int *)(v12 + 8LL * v11);
      sub_1F12210((__int64)a1, *(_DWORD *)(v12 + 8LL * v11), v12, v9, a5, a6);
      v14 = *((_BYTE *)v8 + 4);
      v15 = (_QWORD *)(a1[33] + 112 * v13);
      if ( v14 == 2 )
        break;
      if ( v14 == 4 )
      {
        *v15 = -1;
        goto LABEL_3;
      }
      if ( v14 != 1 )
        goto LABEL_3;
      sub_16AF570(v15 + 1, v10);
      if ( !*((_BYTE *)v8 + 5) )
        goto LABEL_4;
LABEL_10:
      v16 = 2 * *v8 + 1;
      v17 = *(unsigned int *)(*(_QWORD *)(a1[30] + 240LL) + 4 * v16);
      sub_1F12210((__int64)a1, *(_DWORD *)(*(_QWORD *)(a1[30] + 240LL) + 4 * v16), v16, v9, a5, a6);
      v18 = *((_BYTE *)v8 + 5);
      v19 = (_QWORD *)(a1[33] + 112 * v17);
      switch ( v18 )
      {
        case 2:
          sub_16AF570(v19, v10);
          goto LABEL_4;
        case 4:
          *v19 = -1;
          goto LABEL_4;
        case 1:
          v8 += 2;
          sub_16AF570(v19 + 1, v10);
          if ( v8 == v6 )
            return;
          break;
        default:
LABEL_4:
          v8 += 2;
          if ( v8 == v6 )
            return;
          break;
      }
    }
    sub_16AF570(v15, v10);
LABEL_3:
    if ( !*((_BYTE *)v8 + 5) )
      goto LABEL_4;
    goto LABEL_10;
  }
}
