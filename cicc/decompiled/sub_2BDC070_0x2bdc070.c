// Function: sub_2BDC070
// Address: 0x2bdc070
//
void __fastcall sub_2BDC070(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rbx
  char *v6; // r14
  char *v7; // r10
  char v8; // si
  char v9; // r8
  char v10; // di
  char *v11; // rax
  __int16 v12; // dx
  char v13; // cl
  char v14; // dl
  char *v15; // rdi
  char *v16; // r13
  char *v17; // rcx
  char *v18; // rax
  char *v19; // rax
  __int64 i; // rbx
  char *v21; // r14
  char v22; // cl
  __int64 v23; // rbx
  char *v24; // [rsp+0h] [rbp-40h]
  char *v25; // [rsp+8h] [rbp-38h]

  v3 = (__int64)&a2[-a1];
  if ( (__int64)&a2[-a1] <= 16 )
    return;
  v5 = a3;
  v6 = a2;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v25 = (char *)(a1 + 1);
  v24 = (char *)(a1 + 2);
  while ( 2 )
  {
    v8 = *(_BYTE *)(a1 + 1);
    v9 = *(_BYTE *)a1;
    --v5;
    v10 = *(v7 - 1);
    v11 = (char *)(a1 + (__int64)&v7[-a1] / 2);
    v12 = __ROL2__(*(_WORD *)a1, 8);
    v13 = *v11;
    if ( v8 >= *v11 )
    {
      if ( v8 < v10 )
        goto LABEL_7;
      if ( v13 < v10 )
      {
LABEL_18:
        *(_BYTE *)a1 = v10;
        v14 = v9;
        *(v7 - 1) = v9;
        v8 = *(_BYTE *)a1;
        v9 = *(_BYTE *)(a1 + 1);
        goto LABEL_8;
      }
LABEL_23:
      *(_BYTE *)a1 = v13;
      *v11 = v9;
      v9 = *(_BYTE *)(a1 + 1);
      v8 = *(_BYTE *)a1;
      v14 = *(v7 - 1);
      goto LABEL_8;
    }
    if ( v13 < v10 )
      goto LABEL_23;
    if ( v8 < v10 )
      goto LABEL_18;
LABEL_7:
    *(_WORD *)a1 = v12;
    v14 = *(v7 - 1);
LABEL_8:
    v15 = v24;
    v16 = v25;
    v17 = v7;
    while ( 1 )
    {
      v6 = v16;
      if ( v8 > v9 )
        goto LABEL_15;
      v18 = v17 - 1;
      if ( v14 <= v8 )
      {
        --v17;
        if ( v16 >= v18 )
          break;
        goto LABEL_14;
      }
      v19 = v17 - 2;
      do
      {
        v17 = v19;
        v14 = *v19--;
      }
      while ( v14 > v8 );
      if ( v16 >= v17 )
        break;
LABEL_14:
      *v16 = v14;
      v14 = *(v17 - 1);
      *v17 = v9;
      v8 = *(_BYTE *)a1;
LABEL_15:
      v9 = *v15;
      ++v16;
      ++v15;
    }
    sub_2BDC070(v16, v7, v5);
    v3 = (__int64)&v16[-a1];
    if ( (__int64)&v16[-a1] > 16 )
    {
      if ( v5 )
      {
        v7 = v16;
        continue;
      }
LABEL_24:
      for ( i = (v3 - 2) >> 1; ; --i )
      {
        sub_2BDBA30(a1, i, v3, *(_BYTE *)(a1 + i));
        if ( !i )
          break;
      }
      v21 = v6 - 1;
      do
      {
        v22 = *v21;
        v23 = (__int64)&v21[-a1];
        *v21-- = *(_BYTE *)a1;
        sub_2BDBA30(a1, 0, v23, v22);
      }
      while ( v23 > 1 );
    }
    break;
  }
}
