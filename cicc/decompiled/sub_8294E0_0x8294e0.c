// Function: sub_8294E0
// Address: 0x8294e0
//
__int64 __fastcall sub_8294E0(_QWORD *a1, __int64 a2, _QWORD **a3, int *a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  _QWORD *v8; // rax
  int v9; // r12d
  char v10; // al
  __int64 v11; // rdi
  int v12; // eax
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  char v15; // dl
  int v16; // edx
  unsigned int v17; // r10d
  const __m128i *v18; // r10
  __int8 v19; // al
  int v20; // edx
  const __m128i *v21; // r13
  _QWORD *v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r13
  char v27; // dl
  __m128i *v28; // rax
  _QWORD *v29; // rax
  int v30; // eax
  int v31; // eax
  _BOOL4 v32; // eax
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  _QWORD *v39[7]; // [rsp+38h] [rbp-38h] BYREF

  v39[0] = *a3;
  if ( (unsigned int)sub_8D3410(a2) )
  {
    v5 = a2;
    if ( *(_BYTE *)(a2 + 140) == 12 )
    {
      do
        v5 = *(_QWORD *)(v5 + 160);
      while ( *(_BYTE *)(v5 + 140) == 12 );
    }
    else
    {
      v5 = a2;
    }
    v38 = *(_QWORD *)(v5 + 176);
    if ( v38 )
    {
      v7 = 0;
      v6 = 0;
    }
    else
    {
      v6 = 0;
      v38 = -(__int64)(((*(_BYTE *)(v5 + 169) >> 5) ^ 1) & 1);
      v7 = 0;
    }
  }
  else
  {
    v38 = 0;
    v7 = *(_QWORD *)(a2 + 160);
    v6 = **(_QWORD **)(a2 + 168);
  }
  v8 = *a3;
  v39[0] = v8;
  if ( v8 )
  {
    v9 = 0;
    while ( v6 )
    {
      v15 = *(_BYTE *)(v6 + 96);
      if ( (v15 & 1) == 0 )
      {
        v6 = *(_QWORD *)v6;
        goto LABEL_23;
      }
      v10 = *((_BYTE *)v8 + 8);
      if ( v10 == 2 )
      {
        v8 = v39[0];
        if ( v15 >= 0 )
          goto LABEL_52;
        goto LABEL_18;
      }
      v11 = *(_QWORD *)(v6 + 40);
      if ( v15 < 0 )
      {
        v29 = sub_724EF0(v11);
        ++v9;
        *a1 = v29;
        *((_DWORD *)v29 + 9) = v9;
        *(_BYTE *)(*a1 + 33LL) |= 1u;
        *(_QWORD *)(*a1 + 80LL) = *(_QWORD *)(v6 + 120);
        v6 = *(_QWORD *)v6;
        a1 = (_QWORD *)*a1;
        if ( !(v7 | v6) )
        {
          v8 = v39[0];
          v16 = 1;
          v17 = 1;
          goto LABEL_58;
        }
LABEL_23:
        v8 = v39[0];
        if ( !v39[0] )
          goto LABEL_24;
      }
      else
      {
        if ( v10 == 1
          || (v12 = sub_8DBE70(v11), v11 = *(_QWORD *)(v6 + 40), v12)
          || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v11 + 96LL) + 178LL) & 0x40) == 0 )
        {
          v13 = sub_724EF0(v11);
          ++v9;
          *a1 = v13;
          v14 = v39[0];
          *((_DWORD *)v13 + 9) = v9;
          a1 = (_QWORD *)*a1;
          v8 = (_QWORD *)*v14;
          if ( *v14 && *((_BYTE *)v8 + 8) == 3 )
            v8 = (_QWORD *)sub_6BBB10(v14);
          v39[0] = v8;
LABEL_18:
          v6 = *(_QWORD *)v6;
          goto LABEL_19;
        }
        v17 = sub_8294E0(a1, *(_QWORD *)(v6 + 40), v39, a4);
        if ( !v17 )
        {
LABEL_85:
          v8 = v39[0];
          v16 = 0;
          goto LABEL_58;
        }
        v34 = (_QWORD *)*a1;
        v8 = v39[0];
        v6 = *(_QWORD *)v6;
        if ( *a1 )
        {
          do
          {
            a1 = v34;
            v34 = (_QWORD *)*v34;
          }
          while ( v34 );
        }
LABEL_19:
        if ( !v8 )
          goto LABEL_24;
      }
    }
    if ( v7 )
    {
      if ( *((_BYTE *)v8 + 8) == 2 )
      {
        v25 = v8[3];
        if ( !v25 )
        {
LABEL_57:
          v8 = v39[0];
          v16 = 0;
          v17 = 0;
          goto LABEL_58;
        }
        while ( !*(_QWORD *)v7 || v25 != **(_QWORD **)v7 )
        {
          v7 = sub_72FD90(*(_QWORD *)(v7 + 112), 7);
          if ( !v7 )
            goto LABEL_57;
        }
        v8 = (_QWORD *)*v39[0];
        if ( !*v39[0] )
        {
          v39[0] = 0;
          BUG();
        }
        v27 = *((_BYTE *)v8 + 8);
        if ( v27 == 3 )
        {
          v8 = (_QWORD *)sub_6BBB10(v39[0]);
          v27 = *((_BYTE *)v8 + 8);
        }
        v39[0] = v8;
        if ( v27 == 2 )
        {
LABEL_52:
          v16 = 0;
          v17 = 0;
          goto LABEL_58;
        }
        v18 = *(const __m128i **)(v7 + 120);
        v19 = v18[8].m128i_i8[12];
        if ( v19 != 12 )
          goto LABEL_36;
        v20 = 1;
LABEL_29:
        v21 = v18;
        do
        {
          v21 = (const __m128i *)v21[10].m128i_i64[0];
          v19 = v21[8].m128i_i8[12];
        }
        while ( v19 == 12 );
        if ( !v20 )
        {
          if ( *((_BYTE *)v39[0] + 8) != 1 )
            goto LABEL_33;
          goto LABEL_48;
        }
LABEL_36:
        if ( v19 != 8 || *((_BYTE *)v39[0] + 8) != 1 && (*(_BYTE *)(v39[0][3] + 27LL) & 0x10) == 0 )
          goto LABEL_37;
LABEL_67:
        v28 = sub_73C570(v18, 1);
        v18 = (const __m128i *)sub_72D600(v28);
LABEL_37:
        ++v9;
        v22 = sub_724EF0((__int64)v18);
        *a1 = v22;
        v23 = v39[0];
        *((_DWORD *)v22 + 9) = v9;
        a1 = (_QWORD *)*a1;
        v24 = *v23;
        if ( *v23 && *(_BYTE *)(v24 + 8) == 3 )
          v24 = sub_6BBB10(v23);
        v39[0] = (_QWORD *)v24;
LABEL_41:
        if ( v7 )
        {
          if ( (unsigned int)sub_8D3B10(a2) )
            goto LABEL_84;
          v7 = sub_72FD90(*(_QWORD *)(v7 + 112), 7);
          v8 = v39[0];
        }
        else
        {
          v6 = 0;
          v8 = v39[0];
          v38 -= v38 > 0;
        }
        goto LABEL_19;
      }
      v18 = *(const __m128i **)(v7 + 120);
      v19 = v18[8].m128i_i8[12];
      if ( v19 == 12 )
      {
LABEL_28:
        v20 = 0;
        goto LABEL_29;
      }
    }
    else
    {
      if ( !v38 )
      {
LABEL_84:
        v8 = v39[0];
        v16 = 0;
        v17 = 1;
        goto LABEL_58;
      }
      if ( *((_BYTE *)v8 + 8) == 2 )
        goto LABEL_57;
      v18 = (const __m128i *)sub_8D4050(a2);
      v19 = v18[8].m128i_i8[12];
      if ( v19 == 12 )
        goto LABEL_28;
    }
    v21 = v18;
    if ( *((_BYTE *)v39[0] + 8) != 1 )
    {
LABEL_33:
      if ( v19 == 8 )
      {
        if ( v21[10].m128i_i8[8] < 0 )
        {
LABEL_35:
          v19 = v21[8].m128i_i8[12];
          goto LABEL_36;
        }
      }
      else
      {
        v35 = (__int64)v18;
        v30 = sub_8DBE70(v18);
        v18 = (const __m128i *)v35;
        if ( v30 )
          goto LABEL_35;
        v31 = sub_8D3BB0(v21);
        v18 = (const __m128i *)v35;
        if ( !v31 )
          goto LABEL_35;
        if ( (v21[11].m128i_i8[3] & 1) != 0 )
          goto LABEL_35;
        v32 = sub_696450((__int64)v39[0], v35);
        v18 = (const __m128i *)v35;
        if ( v32 )
          goto LABEL_35;
      }
      v17 = sub_8294E0(a1, v21, v39, a4);
      if ( !v17 )
        goto LABEL_85;
      v33 = (_QWORD *)*a1;
      if ( *a1 )
      {
        do
        {
          a1 = v33;
          v33 = (_QWORD *)*v33;
        }
        while ( v33 );
      }
      goto LABEL_41;
    }
LABEL_48:
    if ( v21[8].m128i_i8[12] != 8 )
      goto LABEL_37;
    goto LABEL_67;
  }
LABEL_24:
  v16 = 0;
  v17 = 1;
LABEL_58:
  *a3 = v8;
  *a4 = v16;
  return v17;
}
