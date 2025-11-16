// Function: sub_374B2B0
// Address: 0x374b2b0
//
__int64 __fastcall sub_374B2B0(__int64 a1, unsigned __int8 *a2)
{
  _BYTE *v3; // r14
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // r13d
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 v20; // r13
  __int64 v21; // rax
  _BYTE *v22; // rdx
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // rax
  _BYTE *v34; // rdx
  char *v35; // rax
  size_t v36; // rdx
  __int64 *v37; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-50h] BYREF
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]

  sub_3741110(a1);
  v3 = *(_BYTE **)(a1 + 160);
  v4 = *a2;
  if ( (unsigned int)(v4 - 30) <= 0xA )
  {
    v14 = sub_374ABE0((__int64 *)a1, *((_QWORD *)a2 + 5));
    if ( !(_BYTE)v14 )
    {
      sub_37417A0((_QWORD *)a1, v3);
      return v14;
    }
    v4 = *a2;
  }
  v5 = (unsigned int)(v4 - 34);
  if ( (unsigned __int8)v5 <= 0x33u )
  {
    v6 = 0x8000000000041LL;
    if ( _bittest64(&v6, v5) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v7 = sub_BD2BC0((__int64)a2);
        v9 = v7 + v8;
        if ( (a2[7] & 0x80u) != 0 )
          v9 -= sub_BD2BC0((__int64)a2);
        v10 = v9 >> 4;
        if ( (_DWORD)v10 )
        {
          v11 = 0;
          v12 = 16LL * (unsigned int)v10;
          do
          {
            v13 = 0;
            if ( (a2[7] & 0x80u) != 0 )
              v13 = sub_BD2BC0((__int64)a2);
            if ( *(_DWORD *)(*(_QWORD *)(v13 + v11) + 8LL) != 1 )
              return 0;
            v11 += 16;
          }
          while ( v12 != v11 );
        }
      }
    }
  }
  v16 = *((_QWORD *)a2 + 6);
  v38 = (unsigned __int8 *)v16;
  if ( v16 )
    sub_B96E90((__int64)&v38, v16, 1);
  v17 = 0;
  if ( (a2[7] & 0x20) != 0 )
    v17 = sub_B91C10((__int64)a2, 37);
  v18 = *(_QWORD *)(a1 + 80);
  v39 = v17;
  v40 = 0;
  if ( v18 )
    sub_B91220(a1 + 80, v18);
  v19 = v38;
  *(_QWORD *)(a1 + 80) = v38;
  if ( v19 )
    sub_B976B0((__int64)&v38, v19, a1 + 80);
  *(_QWORD *)(a1 + 88) = v39;
  *(_QWORD *)(a1 + 96) = v40;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 752LL);
  if ( *a2 == 85 )
  {
    v20 = *((_QWORD *)a2 - 4);
    if ( v20 )
    {
      if ( !*(_BYTE *)v20 && *(_QWORD *)(v20 + 24) == *((_QWORD *)a2 + 10) )
      {
        if ( (*(_BYTE *)(v20 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v20 + 7) & 0x10) != 0 )
        {
          v37 = *(__int64 **)(a1 + 144);
          v35 = (char *)sub_BD5D20(*((_QWORD *)a2 - 4));
          if ( (unsigned __int8)sub_980AF0(*v37, v35, v36, &v38) )
          {
            if ( (unsigned __int8)sub_F50940(*(_QWORD **)(a1 + 144), (unsigned int)v38) )
              return 0;
          }
        }
        if ( *(_DWORD *)(v20 + 36) == 354
          && ((unsigned __int8)sub_A747A0((_QWORD *)a2 + 9, "trap-func-name", 0xEu)
           || (unsigned __int8)sub_B49590((__int64)a2, "trap-func-name", 0xEu)) )
        {
          return 0;
        }
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 152) )
  {
    LOBYTE(v32) = sub_3745920((__int64 *)a1, (__int64)a2, *a2 - 29);
    v14 = v32;
    if ( (_BYTE)v32 )
      goto LABEL_43;
    sub_3741080(a1);
    v33 = *(_QWORD *)(a1 + 40);
    v34 = *(_BYTE **)(a1 + 176);
    if ( v34 != *(_BYTE **)(v33 + 752) )
    {
      sub_3741640((_QWORD *)a1, *(_BYTE **)(v33 + 752), v34);
      v33 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 176) = *(_QWORD *)(v33 + 752);
  }
  v14 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(*(_QWORD *)a1 + 24LL))(a1, a2);
  if ( !(_BYTE)v14 )
  {
    sub_3741080(a1);
    v21 = *(_QWORD *)(a1 + 40);
    v22 = *(_BYTE **)(a1 + 176);
    if ( v22 != *(_BYTE **)(v21 + 752) )
      sub_3741640((_QWORD *)a1, *(_BYTE **)(v21 + 752), v22);
    v23 = *(_QWORD *)(a1 + 80);
    v38 = 0;
    v39 = 0;
    v40 = 0;
    if ( v23 )
    {
      sub_B91220(a1 + 80, v23);
      v24 = v38;
      *(_QWORD *)(a1 + 80) = v38;
      if ( v24 )
        sub_B976B0((__int64)&v38, v24, a1 + 80);
      *(_QWORD *)(a1 + 88) = v39;
      *(_QWORD *)(a1 + 96) = v40;
    }
    else
    {
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 96) = 0;
    }
    if ( (unsigned int)*a2 - 30 <= 0xA )
    {
      sub_37417A0((_QWORD *)a1, v3);
      v25 = *(_QWORD *)(a1 + 40);
      v26 = *(_QWORD *)(v25 + 856);
      v27 = *(unsigned int *)(v25 + 880);
      v28 = (*(_QWORD *)(v25 + 864) - v26) >> 4;
      if ( v27 > v28 )
      {
        sub_3741A50((const __m128i **)(v25 + 856), v27 - v28);
        return v14;
      }
      if ( v27 < v28 )
      {
        v29 = v26 + 16 * v27;
        if ( *(_QWORD *)(v25 + 864) != v29 )
        {
          *(_QWORD *)(v25 + 864) = v29;
          return v14;
        }
      }
    }
    return 0;
  }
LABEL_43:
  v30 = *(_QWORD *)(a1 + 80);
  v38 = 0;
  v39 = 0;
  v40 = 0;
  if ( v30 )
  {
    sub_B91220(a1 + 80, v30);
    v31 = v38;
    *(_QWORD *)(a1 + 80) = v38;
    if ( v31 )
      sub_B976B0((__int64)&v38, v31, a1 + 80);
    *(_QWORD *)(a1 + 88) = v39;
    *(_QWORD *)(a1 + 96) = v40;
  }
  else
  {
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 96) = 0;
  }
  return v14;
}
