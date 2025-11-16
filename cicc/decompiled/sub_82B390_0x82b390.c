// Function: sub_82B390
// Address: 0x82b390
//
__int64 __fastcall sub_82B390(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // r14
  char v9; // al
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  char v18; // al
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rdx
  char v23; // al
  char i; // al
  _QWORD *v25; // rax
  _QWORD *v26; // r10
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r11
  __int64 v31; // rbx
  __int64 v32; // r12
  int v33; // eax
  __int64 v34; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  _QWORD *v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __m128i *v38; // [rsp+10h] [rbp-50h]
  __int64 *v39[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a2;
  v7 = a1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(_BYTE *)(v8 + 80);
  if ( v9 == 16 )
  {
    v8 = **(_QWORD **)(v8 + 88);
    v9 = *(_BYTE *)(v8 + 80);
  }
  if ( v9 == 24 )
    v8 = *(_QWORD *)(v8 + 88);
  v10 = *(__int64 **)(a1 + 40);
  if ( !unk_4D04314 || (v39[0] = *(__int64 **)(a1 + 40), !v10) )
  {
    v11 = qword_4F5F830;
    if ( !qword_4F5F830 )
      goto LABEL_25;
    goto LABEL_7;
  }
  v18 = *((_BYTE *)v10 + 8);
  if ( v18 == 3 )
  {
    sub_72F220(v39);
    v10 = v39[0];
    if ( v39[0] )
    {
      v18 = *((_BYTE *)v39[0] + 8);
      goto LABEL_13;
    }
    goto LABEL_24;
  }
LABEL_13:
  a3 = v39[0];
  if ( v18 )
    goto LABEL_14;
LABEL_18:
  v19 = v10[4];
  if ( (unsigned int)sub_8D2E30(v19) )
  {
    v20 = sub_8D46C0(v19);
  }
  else
  {
    v20 = v19;
    if ( (unsigned int)sub_8D3D10(v19) )
      v20 = sub_8D4870(v19);
  }
  while ( *(_BYTE *)(v20 + 140) == 12 )
    v20 = *(_QWORD *)(v20 + 160);
  if ( !(unsigned int)sub_8D2310(v20) )
    goto LABEL_23;
  v21 = **(__int64 ***)(v20 + 168);
  v36 = v21;
  if ( !v21 )
    goto LABEL_23;
  while ( (v21[4] & 0x10) == 0 )
  {
    v21 = (__int64 *)*v21;
    if ( !v21 )
      goto LABEL_23;
  }
  a4 = *(_QWORD **)(v7 + 120);
  if ( !a4 )
  {
LABEL_62:
    a2 = 0;
    a5 = (__int64)sub_73EDA0((const __m128i *)v20, 0);
    goto LABEL_63;
  }
  while ( 1 )
  {
    v22 = a4[12];
    if ( !v22 )
      goto LABEL_36;
    v23 = *(_BYTE *)(v22 + 80);
    if ( v23 == 16 )
    {
      v22 = **(_QWORD **)(v22 + 88);
      v23 = *(_BYTE *)(v22 + 80);
    }
    if ( v23 == 24 )
      v22 = *(_QWORD *)(v22 + 88);
    a5 = *(_QWORD *)(*(_QWORD *)(v22 + 88) + 152LL);
    for ( i = *(_BYTE *)(a5 + 140); i == 12; i = *(_BYTE *)(a5 + 140) )
      a5 = *(_QWORD *)(a5 + 160);
    if ( a5 == v20 )
      break;
    if ( i == 7 && *(_BYTE *)(v20 + 140) == 7 )
    {
      v25 = *(_QWORD **)(a5 + 168);
      v26 = (_QWORD *)*v25;
      if ( *v25 )
      {
        a6 = v36;
        v27 = (_QWORD *)*v25;
        do
        {
          v28 = v27[6];
          if ( v28 )
          {
            do
            {
              a2 = v28;
              v28 = *(_QWORD *)(v28 + 48);
            }
            while ( v28 && a2 != v28 );
            v29 = a6[6];
            if ( v29 )
            {
              do
              {
                v30 = v29;
                v29 = *(_QWORD *)(v29 + 48);
              }
              while ( v29 && v30 != v29 );
              if ( a2 == v30 )
                goto LABEL_58;
            }
          }
          v27 = (_QWORD *)*v27;
          a6 = (_QWORD *)*a6;
        }
        while ( v27 && a6 );
      }
    }
LABEL_36:
    a4 = (_QWORD *)*a4;
    if ( !a4 )
      goto LABEL_62;
  }
  v26 = **(_QWORD ***)(a5 + 168);
  if ( !v26 )
    goto LABEL_23;
LABEL_58:
  v37 = v20;
  v31 = v22;
  v34 = v7;
  v32 = (__int64)v26;
  v35 = a5;
  do
  {
    if ( (*(_BYTE *)(v32 + 32) & 0x10) != 0 )
    {
      a2 = v32;
      sub_895AD0(v31, v32);
    }
    v32 = *(_QWORD *)v32;
  }
  while ( v32 );
  v20 = v37;
  a5 = v35;
  v7 = v34;
LABEL_63:
  if ( a5 == v20 )
  {
LABEL_23:
    a3 = v39[0];
    v10 = (__int64 *)*v39[0];
    v39[0] = v10;
    if ( v10 )
      goto LABEL_15;
    goto LABEL_24;
  }
  v38 = (__m128i *)a5;
  if ( (unsigned int)sub_8D2E30(v19) )
  {
    a2 = 0;
    a5 = sub_72D2E0(v38);
  }
  else
  {
    v33 = sub_8D3D10(v19);
    a5 = (__int64)v38;
    if ( v33 )
    {
      a2 = sub_8D4890(v19);
      a5 = (__int64)sub_73F0A0(v38, a2);
    }
  }
  a3 = v39[0];
  v39[0][4] = a5;
LABEL_14:
  while ( 1 )
  {
    v10 = (__int64 *)*a3;
    v39[0] = v10;
    if ( !v10 )
      break;
LABEL_15:
    v18 = *((_BYTE *)v10 + 8);
    if ( v18 != 3 )
      goto LABEL_13;
    sub_72F220(v39);
    v10 = v39[0];
    if ( !v39[0] )
      break;
    a3 = v39[0];
    if ( !*((_BYTE *)v39[0] + 8) )
      goto LABEL_18;
  }
LABEL_24:
  v11 = qword_4F5F830;
  v10 = *(__int64 **)(v7 + 40);
  if ( qword_4F5F830 )
  {
LABEL_7:
    qword_4F5F830 = *(_QWORD *)v11;
    goto LABEL_8;
  }
LABEL_25:
  v11 = sub_822B10(32, a2, (__int64)a3, (__int64)a4, a5, (__int64)a6);
LABEL_8:
  *(_BYTE *)(v11 + 28) &= ~1u;
  v12 = qword_4F5F838;
  *(_QWORD *)(v11 + 8) = v8;
  *(_QWORD *)(v11 + 16) = v10;
  *(_DWORD *)(v11 + 24) = 0;
  *(_QWORD *)v11 = v12;
  v13 = *(unsigned __int8 *)(v7 + 33);
  qword_4F5F838 = v11;
  result = sub_8B74F0(v8, v7 + 40, v13, v6);
  v15 = (__int64 *)qword_4F5F838;
  qword_4F5F838 = *(_QWORD *)qword_4F5F838;
  v16 = qword_4F5F830;
  qword_4F5F830 = (__int64)v15;
  *v15 = v16;
  *(_QWORD *)(v7 + 8) = result;
  *(_BYTE *)(v7 + 32) = 0;
  if ( (*(_BYTE *)(v7 + 145) & 0x40) != 0 )
  {
    v17 = *(_QWORD *)(result + 88);
    *(_QWORD *)(v7 + 72) = result;
    *(_QWORD *)(v7 + 64) = v17;
  }
  return result;
}
