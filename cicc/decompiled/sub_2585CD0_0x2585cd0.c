// Function: sub_2585CD0
// Address: 0x2585cd0
//
_BOOL8 __fastcall sub_2585CD0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r10
  _BYTE *v6; // rdi
  __int64 v7; // r10
  unsigned __int64 *v8; // r15
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int8 *v11; // r14
  int v12; // edx
  unsigned __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // r11
  _BOOL4 v19; // r12d
  unsigned __int8 *v21; // rdi
  __int64 v22; // rdx
  unsigned int v23; // eax
  unsigned int v24; // edi
  unsigned int v25; // esi
  int v26; // edx
  unsigned int v27; // ecx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // r11
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rbx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 *v37; // rax
  unsigned __int64 v38; // rax
  __int64 v39; // rbx
  unsigned __int64 v40; // rax
  __int64 v41; // [rsp+10h] [rbp-A8h]
  char v42; // [rsp+27h] [rbp-91h]
  unsigned __int64 v44; // [rsp+30h] [rbp-88h]
  char v45; // [rsp+3Fh] [rbp-79h] BYREF
  __int64 v46; // [rsp+40h] [rbp-78h] BYREF
  _QWORD *v47; // [rsp+48h] [rbp-70h] BYREF
  __int64 v48; // [rsp+50h] [rbp-68h]
  _BYTE v49[96]; // [rsp+58h] [rbp-60h] BYREF

  v3 = (_QWORD *)(a1 + 72);
  v4 = *(_QWORD *)(a2 + 208);
  v45 = 0;
  v41 = *(_QWORD *)(v4 + 104);
  v47 = v49;
  v48 = 0x300000000LL;
  v42 = sub_2526B50(a2, (const __m128i *)(a1 + 72), a1, (__int64)&v47, 3u, &v45, 1u);
  if ( v42 )
  {
    v5 = (unsigned int)v48;
    if ( (_DWORD)v48 == 1 )
    {
      v39 = *v47;
      v40 = sub_250D070(v3);
      v5 = (unsigned int)v48;
      v42 = v39 != v40;
    }
  }
  else
  {
    v31 = sub_250D070(v3);
    v34 = sub_2509740(v3);
    v35 = (unsigned int)v48;
    v36 = (unsigned int)v48 + 1LL;
    if ( v36 > HIDWORD(v48) )
    {
      sub_C8D5F0((__int64)&v47, v49, v36, 0x10u, v32, v33);
      v35 = (unsigned int)v48;
    }
    v37 = &v47[2 * v35];
    *v37 = v31;
    v37[1] = v34;
    v5 = (unsigned int)(v48 + 1);
    LODWORD(v48) = v48 + 1;
  }
  v6 = v47;
  v7 = 2 * v5;
  v8 = &v47[v7];
  if ( &v47[v7] == v47 )
  {
    v29 = 0x100000000LL;
    goto LABEL_37;
  }
  v9 = v47;
  v44 = 1;
  v10 = 0x100000000LL;
  do
  {
    v11 = (unsigned __int8 *)*v9;
    v12 = *(unsigned __int8 *)*v9;
    if ( (unsigned int)(v12 - 12) <= 1 || (_BYTE)v12 == 20 )
      goto LABEL_6;
    v13 = sub_250D2C0(*v9, 0);
    v15 = sub_2584D90(a2, v13, v14, a1, 0, 0, 1);
    if ( !v15 || a1 == v15 && v42 != 1 )
    {
      v21 = sub_25536C0((__int64)v11, &v46, v41, 1);
      if ( !v21 )
      {
        v38 = (unsigned int)(1LL << sub_BD5420(v11, v41));
        if ( v44 >= v38 )
          v38 = v44;
        v44 = v38;
        goto LABEL_33;
      }
      v22 = 1LL << sub_BD5420(v21, v41);
      v23 = v22;
      v24 = abs32(v46);
      if ( (_DWORD)v46 )
      {
        v25 = v24 & (v22 - 1);
        if ( (_DWORD)v22 )
        {
          while ( v25 )
          {
            v26 = v23 % v25;
            v23 = v25;
            v25 = v26;
          }
        }
        else
        {
          v23 = v24;
        }
      }
      else if ( !(_DWORD)v22 )
      {
LABEL_33:
        v10 = v44;
        goto LABEL_19;
      }
      _BitScanReverse(&v27, v23);
      v28 = 0x80000000 >> (v27 ^ 0x1F);
      if ( v44 >= v28 )
        v28 = v44;
      v44 = v28;
      goto LABEL_33;
    }
    v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL);
    if ( v16 == sub_2534F40 )
      v17 = v15 + 88;
    else
      v17 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v16)(v15, v13);
    v18 = v10;
    if ( *(_QWORD *)(v17 + 16) <= v10 )
      v18 = *(_QWORD *)(v17 + 16);
    if ( v18 < v44 )
      v18 = v44;
    v10 = v18;
LABEL_19:
    if ( v10 == 1 )
    {
      v6 = v47;
      v19 = 0;
      *(_QWORD *)(a1 + 104) = *(_QWORD *)(a1 + 96);
      goto LABEL_21;
    }
LABEL_6:
    v9 += 2;
  }
  while ( v8 != v9 );
  v6 = v47;
  v29 = v10;
LABEL_37:
  v30 = *(_QWORD *)(a1 + 104);
  if ( v30 <= v29 )
    v29 = *(_QWORD *)(a1 + 104);
  if ( v29 < *(_QWORD *)(a1 + 96) )
    v29 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 104) = v29;
  v19 = v30 == v29;
LABEL_21:
  if ( v6 != v49 )
    _libc_free((unsigned __int64)v6);
  return v19;
}
