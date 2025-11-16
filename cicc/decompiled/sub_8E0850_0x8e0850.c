// Function: sub_8E0850
// Address: 0x8e0850
//
__int64 __fastcall sub_8E0850(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v4; // r8
  unsigned int v5; // r14d
  __int64 v6; // r12
  __int64 v7; // rbx
  char v8; // r15
  int v9; // eax
  bool v10; // r15
  __int64 v11; // rcx
  __int64 v13; // rdx
  __int64 v14; // r12
  __int128 *v15; // rsi
  __int128 *v16; // rdi
  __int64 v17; // rax
  __int64 i; // r13
  __int64 v19; // r12
  char v20; // al
  __int64 v21; // r12
  __int64 v22; // r13
  char v23; // al
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  unsigned int v35; // [rsp+38h] [rbp-38h]
  char v36; // [rsp+3Dh] [rbp-33h]
  char v37; // [rsp+3Eh] [rbp-32h]
  bool v38; // [rsp+3Fh] [rbp-31h]

  v4 = a3;
  v5 = 0;
  v6 = a2;
  v7 = a1;
  v8 = *(_BYTE *)(a3 + 122);
  *a4 = 0;
  v9 = *(unsigned __int8 *)(a1 + 80);
  v10 = (v8 & 4) != 0;
  if ( (_BYTE)v9 != 17 )
    goto LABEL_4;
  v7 = *(_QWORD *)(a1 + 88);
  v5 = 1;
  v9 = *(unsigned __int8 *)(v7 + 80);
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
LABEL_4:
      ;
    }
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  v11 = *(_QWORD *)(v6 + 168);
  v38 = v10;
  v31 = v6;
  v34 = v11;
  v33 = *(_QWORD *)(v11 + 40);
  v35 = *(_BYTE *)(v11 + 18) & 0x7F;
  v37 = *(_BYTE *)(v11 + 19) >> 6;
  while ( 1 )
  {
    v13 = v9 & 0xFFFFFFF7;
    if ( (v9 & 0xF7) == 0x10 )
      goto LABEL_6;
    if ( v38 || (_BYTE)v9 == 20 )
      goto LABEL_6;
    v14 = *(_QWORD *)(v7 + 88);
    v15 = *(__int128 **)(a3 + 400);
    v16 = *(__int128 **)(v14 + 216);
    if ( v15 != v16 && !(unsigned int)sub_739400(v16, v15) )
      goto LABEL_6;
    v17 = 0;
    if ( (*(_BYTE *)(v14 + 194) & 0x40) != 0 )
      v17 = *(_QWORD *)(v14 + 232);
    if ( *(_QWORD *)(a3 + 320) != v17 )
      goto LABEL_6;
    for ( i = *(_QWORD *)(v14 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v19 = *(_QWORD *)(i + 168);
    v30 = *(_QWORD *)(v19 + 40);
    v20 = *(_BYTE *)(v19 + 19) >> 6;
    v36 = v20;
    if ( v37 == v20 )
    {
      v13 = v35;
      LOBYTE(v11) = (*(_BYTE *)(v19 + 18) & 0x7F) != 0;
      if ( (_BYTE)v11 != (v35 != 0) )
      {
        v13 = (__int64)&qword_4D0495C;
        v11 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
        if ( qword_4D0495C )
          goto LABEL_6;
      }
      if ( v30
        && v33
        && (v35 != (*(_BYTE *)(v19 + 18) & 0x7F) || v33 != v30 && !(unsigned int)sub_8D97D0(v30, v33, 0, v11, v4)) )
      {
        goto LABEL_6;
      }
    }
    else if ( v20 && v37 )
    {
      goto LABEL_6;
    }
    if ( ((*(_BYTE *)(v34 + 16) ^ *(_BYTE *)(v19 + 16)) & 1) != 0 )
      goto LABEL_6;
    v11 = *(_QWORD *)v19;
    if ( !(*(_QWORD *)v34 | *(_QWORD *)v19) )
      break;
    if ( *(_QWORD *)v19 && *(_QWORD *)v34 )
    {
      v29 = v19;
      v21 = *(_QWORD *)v19;
      v28 = i;
      v22 = *(_QWORD *)v34;
      do
      {
        v23 = *(_BYTE *)(v22 + 32) ^ *(_BYTE *)(v21 + 32);
        if ( (v23 & 0x40) != 0
          || v23 < 0
          || ((*(_BYTE *)(v22 + 33) ^ *(_BYTE *)(v21 + 33)) & 1) != 0
          || !(unsigned int)sub_8DED30(*(_QWORD *)(v21 + 8), *(_QWORD *)(v22 + 8), 4096, v11, v4) )
        {
          break;
        }
        v21 = *(_QWORD *)v21;
        v22 = *(_QWORD *)v22;
        if ( !(v22 | v21) )
        {
          v19 = v29;
          i = v28;
          goto LABEL_34;
        }
      }
      while ( v21 && v22 );
    }
LABEL_6:
    if ( !v5 )
      return 1;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return v5;
    v9 = *(unsigned __int8 *)(v7 + 80);
  }
LABEL_34:
  if ( ((*(_BYTE *)(v19 + 20) & 2) != 0 || (*(_BYTE *)(v34 + 20) & 2) != 0) && !sub_8D73A0(v31, i) )
    goto LABEL_6;
  if ( (v30 == 0) == (v33 == 0) )
  {
    if ( v37 == v36 )
    {
      if ( dword_4F06978
        && ((LOBYTE(v13) = v30 == 0, sub_8DADD0(i, v31, v13, v11, v4)) || sub_8DADD0(v31, i, v25, v26, v27)) )
      {
        v5 = 0;
        *a4 = 2862;
      }
      else
      {
        v5 = 0;
        *a4 = 311;
      }
    }
    else
    {
      v5 = 0;
      *a4 = 2449;
    }
  }
  else
  {
    v5 = 0;
    *a4 = 751;
  }
  return v5;
}
