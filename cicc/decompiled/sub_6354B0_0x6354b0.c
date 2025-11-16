// Function: sub_6354B0
// Address: 0x6354b0
//
__int64 __fastcall sub_6354B0(__int64 a1, __m128i *a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r8
  unsigned __int8 v7; // r9
  int *v8; // rax
  __int64 v9; // rbx
  __int8 v10; // al
  __int64 v11; // r10
  __int64 i; // rdx
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // r15
  char v16; // dl
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // r10
  char v27; // cl
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+0h] [rbp-60h]
  __int64 v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  int v46; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v47; // [rsp+20h] [rbp-40h] BYREF
  __int64 v48[7]; // [rsp+28h] [rbp-38h] BYREF

  v47 = 0;
  if ( dword_4D04428 )
  {
    if ( (_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A0 > 0x7723u )
      {
LABEL_5:
        v4 = sub_6E2F40(1);
        v48[0] = v4;
        v5 = v4;
        *(_QWORD *)(v4 + 32) = *a3;
        *(_QWORD *)(v4 + 40) = *a3;
        sub_634B10(v48, a1, 0, a2, (__int64)a3, &v47);
        sub_6E1990(v5);
        return v47;
      }
    }
    else if ( !dword_4F077BC || qword_4F077A8 > 0x9EFBu )
    {
      goto LABEL_5;
    }
  }
  v7 = a2[2].m128i_u8[8];
  v8 = &v46;
  v46 = 0;
  if ( (v7 & 0x20) == 0 )
    v8 = 0;
  v9 = sub_87CAB0(a1, (_DWORD)a3, a1, 0, 1, ((v7 >> 6) ^ 1) & 1, 1, (__int64)v8, 0);
  if ( v46 )
    a2[2].m128i_i8[9] |= 2u;
  v10 = a2[2].m128i_i8[8];
  v11 = 0;
  if ( dword_4D048B8 && (v10 & 4) == 0 )
  {
    for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 24LL);
    if ( v11 )
    {
      v13 = *(_QWORD *)(v11 + 88);
      if ( (*(_BYTE *)(v13 + 194) & 8) == 0 || (v11 = 0, (*(_BYTE *)(v13 + 206) & 0x10) != 0) )
      {
        v11 = sub_630000(a1, a2, (int)a3);
        v10 = a2[2].m128i_i8[8];
      }
    }
  }
  v14 = v10 & 0x40;
  v15 = 0;
  if ( !v9 || (a2[2].m128i_i8[9] & 2) != 0 )
  {
    if ( !v14 )
    {
      v42 = v11;
      v35 = sub_725A70(1);
      v11 = v42;
      v15 = v35;
      v14 = a2[2].m128i_i8[8] & 0x40;
    }
    a2[2].m128i_i8[9] |= 4u;
  }
  else
  {
    if ( !v14 )
    {
      v41 = v11;
      v19 = sub_62FD00(v9, 0, (((unsigned __int8)a2[2].m128i_i8[10] >> 5) ^ 1) & 1, 0);
      v11 = v41;
      *(_BYTE *)(v19 + 72) |= 4u;
      v15 = v19;
      v24 = a2[2].m128i_i8[8];
      if ( (v24 & 4) != 0 )
      {
        if ( *(_BYTE *)(v15 + 48) == 2 )
        {
          v36 = *(_QWORD *)(v15 + 56);
        }
        else
        {
          v37 = sub_724D50(0);
          if ( (*(_BYTE *)(*(_QWORD *)(v15 + 56) + 193LL) & 2) != 0 )
          {
            v39 = v41;
            v44 = v37;
            v38 = sub_71AAF0(v15, 1, 1, 1, a3, v37);
            v36 = v44;
            v11 = v39;
            if ( v38 )
            {
              v24 = a2[2].m128i_i8[8];
            }
            else
            {
              sub_72C970(v44);
              v24 = a2[2].m128i_i8[8];
              v36 = v44;
              v11 = v39;
            }
          }
          else
          {
            v40 = v41;
            v45 = v37;
            sub_685360(2401, a3);
            sub_72C970(v45);
            v24 = a2[2].m128i_i8[8];
            v11 = v40;
            v36 = v45;
          }
        }
        v47 = v36;
        v14 = v24 & 0x40;
      }
      else
      {
        v25 = sub_724DC0(v9, 0, v20, v21, v22, v23);
        v26 = v41;
        v48[0] = v25;
        v27 = *(_BYTE *)(v9 + 193);
        if ( (v27 & 2) != 0 && (v28 = sub_71AAF0(v15, 1, 0, (v27 & 4) != 0, a3, v25), v26 = v41, v28) )
        {
          if ( (*(_BYTE *)(v48[0] + 170) & 0x40) != 0 )
            a2[2].m128i_i8[9] |= 0x10u;
          v32 = sub_724E50(v48, 1, v29, v30, v31);
          v11 = v41;
          v47 = v32;
          if ( v41 )
          {
            v33 = sub_725A70(2);
            v11 = v41;
            v15 = v33;
            v34 = v47;
            *(_QWORD *)(v15 + 56) = v47;
            if ( (*(_BYTE *)(v34 + 170) & 0x40) != 0 )
              *(_BYTE *)(v15 + 50) |= 0x80u;
            v47 = 0;
          }
        }
        else
        {
          v43 = v26;
          sub_724E30(v48);
          v11 = v43;
        }
        v14 = a2[2].m128i_i8[8] & 0x40;
      }
    }
    v16 = *(_BYTE *)(v9 + 193);
    if ( (v16 & 2) == 0 )
    {
      a2[2].m128i_i8[10] |= 0x80u;
      v16 = *(_BYTE *)(v9 + 193);
    }
    if ( (v16 & 0x10) != 0 && (*(_BYTE *)(a1 + 179) & 4) != 0 )
      a2[2].m128i_i8[9] |= 0x10u;
  }
  if ( v11 )
  {
    if ( v14 )
      return v47;
    *(_QWORD *)(v15 + 16) = v11;
    if ( (a2[2].m128i_i8[10] & 0x20) != 0 )
    {
      if ( !dword_4D048B8 )
      {
LABEL_36:
        v14 = a2[2].m128i_i8[8] & 0x40;
        goto LABEL_37;
      }
    }
    else
    {
      *(_BYTE *)(v11 + 193) |= 0x40u;
      if ( !dword_4D048B8 )
        goto LABEL_36;
    }
    sub_734250(v15, (((unsigned __int8)a2[2].m128i_i8[10] >> 4) ^ 1) & 1);
    goto LABEL_36;
  }
LABEL_37:
  v6 = v47;
  if ( !v14 && !v47 )
  {
    v18 = sub_724D50(9);
    *(_QWORD *)(v18 + 176) = v15;
    v6 = v18;
    *(_QWORD *)(v18 + 128) = a1;
    a2[2].m128i_i8[9] |= 4u;
  }
  return v6;
}
