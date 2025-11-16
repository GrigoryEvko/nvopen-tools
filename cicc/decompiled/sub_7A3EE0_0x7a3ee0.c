// Function: sub_7A3EE0
// Address: 0x7a3ee0
//
__int64 __fastcall sub_7A3EE0(__int64 a1, __int64 a2, __int64 *a3, _BYTE **a4, __int64 a5, __int64 a6)
{
  _BYTE *v9; // rbx
  __int64 v10; // rax
  FILE *v11; // rcx
  __int64 i; // r12
  __int64 v13; // rax
  FILE *v14; // rcx
  char v15; // al
  const __m128i **j; // r10
  unsigned int v17; // r12d
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rax
  char v22; // dl
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 **v26; // rax
  FILE *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  FILE *v29; // [rsp+18h] [rbp-58h]
  const __m128i **v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]

  v9 = *a4;
  if ( **a4 == 48 )
  {
    v19 = *((_QWORD *)v9 + 1);
    v20 = *(_BYTE *)(v19 + 8);
    if ( v20 == 1 )
    {
      *v9 = 2;
      *((_QWORD *)v9 + 1) = *(_QWORD *)(v19 + 32);
    }
    else if ( v20 == 2 )
    {
      *v9 = 59;
      *((_QWORD *)v9 + 1) = *(_QWORD *)(v19 + 32);
    }
    else
    {
      if ( v20 )
        sub_721090();
      *v9 = 6;
      *((_QWORD *)v9 + 1) = *(_QWORD *)(v19 + 32);
    }
  }
  v10 = sub_773040(v9);
  v11 = (FILE *)((char *)a3 + 28);
  if ( v10 )
  {
    for ( i = *a3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v29 = v11;
    v33 = 0;
    v13 = sub_823970(0);
    v32 = 0;
    v14 = v29;
    v31 = v13;
    v15 = *v9;
    if ( *v9 == 7 )
    {
      j = (const __m128i **)&v31;
      v26 = *(__int64 ***)(*((_QWORD *)v9 + 1) + 216LL);
      if ( !v26 )
        goto LABEL_8;
      v23 = *v26;
    }
    else if ( v15 == 11 )
    {
      v23 = *(__int64 **)(*((_QWORD *)v9 + 1) + 240LL);
    }
    else
    {
      j = (const __m128i **)&v31;
      if ( v15 != 6 )
      {
LABEL_8:
        v17 = sub_77AFD0(a1, i, j, v14, a5, a6);
        sub_823A00(v31, 24 * v32);
        return v17;
      }
      v21 = *((_QWORD *)v9 + 1);
      v22 = *(_BYTE *)(v21 + 140);
      if ( (unsigned __int8)(v22 - 9) <= 2u )
      {
        v23 = *(__int64 **)(*(_QWORD *)(v21 + 168) + 168LL);
      }
      else
      {
        if ( v22 != 12 )
          goto LABEL_8;
        v23 = **(__int64 ***)(v21 + 168);
      }
    }
    for ( j = (const __m128i **)&v31; v23; v23 = (__int64 *)*v23 )
    {
      if ( *((_BYTE *)v23 + 8) <= 2u )
      {
        v24 = v33;
        if ( v33 == v32 )
        {
          v27 = v14;
          v28 = v33;
          v30 = j;
          sub_7A3E20(j);
          v14 = v27;
          v24 = v28;
          j = v30;
        }
        v25 = v31 + 24 * v24;
        if ( v25 )
        {
          *(_BYTE *)v25 = 48;
          *(_QWORD *)(v25 + 8) = v23;
          *(_DWORD *)(v25 + 16) = 0;
        }
        v33 = v24 + 1;
      }
    }
    goto LABEL_8;
  }
  v17 = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Du, v11, (_QWORD *)(a1 + 96));
    sub_770D30(a1);
  }
  return v17;
}
