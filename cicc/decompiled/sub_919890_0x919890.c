// Function: sub_919890
// Address: 0x919890
//
__int64 __fastcall sub_919890(__int64 a1, __int64 a2)
{
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r8
  int v9; // r14d
  __int64 *v10; // rdx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 v14; // r13
  __int64 *v15; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 *v21; // rdx
  unsigned __int8 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  _QWORD *v26; // rsi
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rcx
  unsigned int i; // eax
  __int64 v31; // rsi
  __int64 v32; // rax
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rax
  _QWORD *v36; // rax
  int v37; // eax
  __int64 v38; // r8
  unsigned int v39; // eax
  __int64 v40; // rdi
  int v41; // r10d
  __int64 *v42; // r9
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  int v46; // r9d
  unsigned int v47; // r13d
  __int64 *v48; // r8
  unsigned int v49; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v50; // [rsp+10h] [rbp-50h] BYREF
  __int64 v51; // [rsp+18h] [rbp-48h]
  __int64 v52; // [rsp+20h] [rbp-40h]

  v4 = *(_BYTE *)(a2 + 140) == 12;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  if ( v4 )
  {
    v5 = a2;
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
    if ( v5 != a2 )
      sub_91B8A0("error while translating tag type!");
  }
  v6 = *(unsigned int *)(a1 + 48);
  v7 = a1 + 24;
  if ( !(_DWORD)v6 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_61;
  }
  v8 = *(_QWORD *)(a1 + 32);
  v9 = 1;
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
  {
LABEL_8:
    v14 = v12[1];
    v15 = v12 + 1;
    if ( v14 )
      goto LABEL_9;
    goto LABEL_53;
  }
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v10 )
      v10 = v12;
    v11 = (v6 - 1) & (v9 + v11);
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
      goto LABEL_8;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v33 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v34 = v33 + 1;
  if ( 4 * (v33 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_61:
    sub_9189C0(v7, 2 * v6);
    v37 = *(_DWORD *)(a1 + 48);
    if ( v37 )
    {
      v6 = (unsigned int)(v37 - 1);
      v38 = *(_QWORD *)(a1 + 32);
      v39 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = *(_DWORD *)(a1 + 40) + 1;
      v10 = (__int64 *)(v38 + 16LL * v39);
      v40 = *v10;
      if ( *v10 != a2 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -4096 )
        {
          if ( v40 == -8192 && !v42 )
            v42 = v10;
          v39 = v6 & (v41 + v39);
          v10 = (__int64 *)(v38 + 16LL * v39);
          v40 = *v10;
          if ( *v10 == a2 )
            goto LABEL_50;
          ++v41;
        }
        if ( v42 )
          v10 = v42;
      }
      goto LABEL_50;
    }
    goto LABEL_84;
  }
  if ( (int)v6 - *(_DWORD *)(a1 + 44) - v34 <= (unsigned int)v6 >> 3 )
  {
    sub_9189C0(v7, v6);
    v43 = *(_DWORD *)(a1 + 48);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 32);
      v46 = 1;
      v47 = v44 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v48 = 0;
      v34 = *(_DWORD *)(a1 + 40) + 1;
      v10 = (__int64 *)(v45 + 16LL * v47);
      v6 = *v10;
      if ( *v10 != a2 )
      {
        while ( v6 != -4096 )
        {
          if ( v6 == -8192 && !v48 )
            v48 = v10;
          v47 = v44 & (v46 + v47);
          v10 = (__int64 *)(v45 + 16LL * v47);
          v6 = *v10;
          if ( *v10 == a2 )
            goto LABEL_50;
          ++v46;
        }
        if ( v48 )
          v10 = v48;
      }
      goto LABEL_50;
    }
LABEL_84:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_50:
  *(_DWORD *)(a1 + 40) = v34;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 44);
  *v10 = a2;
  v15 = v10 + 1;
  v10[1] = 0;
LABEL_53:
  v35 = sub_BCC900(**(_QWORD **)a1, v6, v10);
  *v15 = v35;
  v14 = v35;
LABEL_9:
  if ( (*(_BYTE *)(v14 + 9) & 1) != 0 || sub_8D23B0(a2) )
    goto LABEL_10;
  if ( !(unsigned __int8)sub_918070(a1, a2, v17, v18) )
  {
    v32 = *(unsigned int *)(a1 + 224);
    if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 228) )
    {
      sub_C8D5F0(a1 + 216, a1 + 232, v32 + 1, 8);
      v32 = *(unsigned int *)(a1 + 224);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v32) = a2;
    ++*(_DWORD *)(a1 + 224);
    goto LABEL_10;
  }
  if ( !*(_BYTE *)(a1 + 180) )
    goto LABEL_55;
  v19 = *(__int64 **)(a1 + 160);
  v20 = *(unsigned int *)(a1 + 172);
  v21 = &v19[v20];
  if ( v19 != v21 )
  {
    while ( *v19 != a2 )
    {
      if ( v21 == ++v19 )
        goto LABEL_54;
    }
    goto LABEL_20;
  }
LABEL_54:
  if ( (unsigned int)v20 < *(_DWORD *)(a1 + 168) )
  {
    *(_DWORD *)(a1 + 172) = v20 + 1;
    *v21 = a2;
    ++*(_QWORD *)(a1 + 152);
  }
  else
  {
LABEL_55:
    sub_C8CC70(a1 + 152, a2);
  }
LABEL_20:
  v22 = sub_91B7B0(a2);
  if ( HIDWORD(qword_4F077B4) && qword_4F077A8 <= 0x9DCFu && v22 && (unsigned __int8)sub_917A60(*(_QWORD *)(a2 + 160)) )
    sub_91B8A0("Bitfields and field types containing bitfields are not supported in packed structures and unions for devi"
               "ce compilation, when using this host compiler!");
  if ( *(_BYTE *)(a2 + 140) == 11 )
  {
    sub_91A3C0(a1, a2, &v50);
  }
  else
  {
    v49 = 0;
    sub_919130((_QWORD *)a1, a2, (__int64)&v50, &v49);
  }
  if ( *(_BYTE *)(a1 + 180) )
  {
    v26 = *(_QWORD **)(a1 + 160);
    v27 = &v26[*(unsigned int *)(a1 + 172)];
    v28 = v26;
    if ( v26 != v27 )
    {
      while ( *v28 != a2 )
      {
        if ( v27 == ++v28 )
          goto LABEL_33;
      }
      v29 = (unsigned int)(*(_DWORD *)(a1 + 172) - 1);
      *(_DWORD *)(a1 + 172) = v29;
      *v28 = v26[v29];
      ++*(_QWORD *)(a1 + 152);
    }
  }
  else
  {
    v36 = (_QWORD *)sub_C8CA60(a1 + 152, a2, v23, v24, v25);
    if ( v36 )
    {
      *v36 = -2;
      ++*(_DWORD *)(a1 + 176);
      ++*(_QWORD *)(a1 + 152);
    }
  }
LABEL_33:
  sub_BD0B50(v14, v50, (v51 - v50) >> 3, v22);
  if ( *(_DWORD *)(a1 + 172) == *(_DWORD *)(a1 + 176) )
  {
    for ( i = *(_DWORD *)(a1 + 224); i; i = *(_DWORD *)(a1 + 224) )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8LL * i - 8);
      *(_DWORD *)(a1 + 224) = i - 1;
      sub_919890(a1, v31);
    }
  }
LABEL_10:
  if ( v50 )
    j_j___libc_free_0(v50, v52 - v50);
  return v14;
}
