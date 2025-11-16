// Function: sub_24E64B0
// Address: 0x24e64b0
//
__int64 __fastcall sub_24E64B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r15
  unsigned __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // r15
  __int64 v12; // rbx
  unsigned int *v13; // r12
  __int64 v14; // rdx
  unsigned int v15; // esi
  void *v16; // rbx
  bool v17; // sf
  __int64 v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rsi
  _BYTE *v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rax
  __int16 v32; // dx
  char v33; // bl
  __int16 v34; // cx
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // r8
  unsigned __int64 v38; // rsi
  unsigned int *v39; // rax
  int v40; // ecx
  unsigned int *v41; // rdx
  __int64 v42; // rcx
  int v43; // eax
  __int64 *v44; // rbx
  __int64 *v45; // rax
  unsigned __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 *v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rsi
  __int64 v53; // [rsp+18h] [rbp-D8h]
  _BYTE *v55; // [rsp+18h] [rbp-D8h]
  __int64 v56; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v58; // [rsp+30h] [rbp-C0h]
  __int64 v59; // [rsp+30h] [rbp-C0h]
  __int64 v60; // [rsp+38h] [rbp-B8h]
  int v61; // [rsp+38h] [rbp-B8h]
  __int64 v62[2]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD v63[4]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v64; // [rsp+80h] [rbp-70h]
  __int64 v65[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v66; // [rsp+B0h] [rbp-40h]

  v3 = a2;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v60 = *(_QWORD *)(a2 + 32 * (2 - v4));
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_QWORD *)(a2 + 32 * (1 - v4));
  sub_D5F1F0(a1, a2);
  v65[0] = *(_QWORD *)(a2 + 72);
  v7 = (unsigned __int64 *)sub_BD5C60(a2);
  v58 = sub_A786C0(v65, v7, 3);
  if ( *(_BYTE *)a2 == 34 )
  {
    v8 = *(_QWORD *)(a2 - 64);
    v62[0] = v5;
    v9 = 0;
    v64 = 257;
    v53 = v8;
    v10 = *(_QWORD *)(a2 - 96);
    v62[1] = v6;
    v56 = v10;
    if ( v60 )
      v9 = *(_QWORD *)(v60 + 24);
    v66 = 257;
    v11 = (__int64)sub_BD2CC0(88, 5u);
    if ( v11 )
    {
      sub_B44260(v11, **(_QWORD **)(v9 + 16), 5, 5u, 0, 0);
      *(_QWORD *)(v11 + 72) = 0;
      sub_B4A9C0(v11, v9, v60, v56, v53, (__int64)v65, v62, 2, 0, 0);
    }
    if ( *(_BYTE *)(a1 + 108) )
    {
      v50 = (__int64 *)sub_BD5C60(v11);
      *(_QWORD *)(v11 + 72) = sub_A7A090((__int64 *)(v11 + 72), v50, -1, 72);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v11,
      v63,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    if ( *(_QWORD *)a1 != *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8) )
    {
      v12 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      v13 = *(unsigned int **)a1;
      do
      {
        v14 = *((_QWORD *)v13 + 1);
        v15 = *v13;
        v13 += 4;
        sub_B99FD0(v11, v15, v14);
      }
      while ( (unsigned int *)v12 != v13 );
      v3 = a2;
    }
    v16 = 0;
    v17 = *(char *)(v11 + 7) < 0;
    *(_WORD *)(v11 + 2) = *(_WORD *)(v3 + 2) & 0xFFC | *(_WORD *)(v11 + 2) & 0xF003;
    if ( v17 )
      v16 = (void *)sub_BD2BC0(v11);
    if ( *(char *)(v3 + 7) < 0 )
    {
      v18 = sub_BD2BC0(v3);
      v20 = 0;
      v21 = (_BYTE *)(v18 + v19);
      if ( *(char *)(v3 + 7) < 0 )
      {
        v55 = (_BYTE *)(v18 + v19);
        v22 = sub_BD2BC0(v3);
        v21 = v55;
        v20 = (_BYTE *)v22;
      }
      if ( v20 != v21 )
        memmove(v16, v20, v21 - v20);
    }
    v23 = (__int64 *)(v11 + 48);
    *(_QWORD *)(v11 + 72) = v58;
    v24 = *(_QWORD *)(v3 + 48);
    v65[0] = v24;
    if ( v24 )
    {
      sub_B96E90((__int64)v65, v24, 1);
      if ( v23 == v65 )
        goto LABEL_21;
      goto LABEL_38;
    }
    if ( v23 == v65 )
      goto LABEL_28;
LABEL_24:
    v25 = *(_QWORD *)(v11 + 48);
    if ( !v25 )
      goto LABEL_28;
    goto LABEL_25;
  }
  if ( *(_BYTE *)a2 != 85 )
    BUG();
  v63[0] = v5;
  v29 = 0;
  v66 = 257;
  v63[1] = v6;
  if ( v60 )
    v29 = *(_QWORD *)(v60 + 24);
  v11 = sub_921880((unsigned int **)a1, v29, v60, (int)v63, 2, (__int64)v65, 0);
  *(_QWORD *)(v11 + 72) = v58;
  v30 = *(_QWORD *)(v3 + 48);
  v65[0] = v30;
  if ( !v30 )
  {
    v23 = (__int64 *)(v11 + 48);
    if ( (__int64 *)(v11 + 48) == v65 )
      goto LABEL_28;
    goto LABEL_24;
  }
  v23 = (__int64 *)(v11 + 48);
  sub_B96E90((__int64)v65, v30, 1);
  if ( (__int64 *)(v11 + 48) == v65 )
  {
LABEL_21:
    if ( v65[0] )
      sub_B91220((__int64)v65, v65[0]);
    goto LABEL_28;
  }
LABEL_38:
  v25 = *(_QWORD *)(v11 + 48);
  if ( v25 )
LABEL_25:
    sub_B91220((__int64)v23, v25);
  v26 = (unsigned __int8 *)v65[0];
  *(_QWORD *)(v11 + 48) = v65[0];
  if ( v26 )
    sub_B976B0((__int64)v65, v26, (__int64)v23);
LABEL_28:
  v27 = *(_QWORD *)(v3 - 32);
  if ( !v27 || *(_BYTE *)v27 || *(_QWORD *)(v27 + 24) != *(_QWORD *)(v3 + 80) )
    BUG();
  if ( *(_DWORD *)(v27 + 36) == 37 )
  {
    if ( *(_BYTE *)v3 != 34 )
    {
LABEL_50:
      sub_24F30B0(v63, *(_QWORD *)(v60 + 40));
      v42 = *(_QWORD *)(a1 + 56);
      if ( v42 )
        v42 -= 24;
      v43 = sub_24F3110(v63, v11, 0, v42);
      v44 = *(__int64 **)(a1 + 72);
      v61 = v43;
      v65[0] = sub_BCE3C0(v44, 0);
      v45 = (__int64 *)sub_BCB120(v44);
      v46 = sub_BCF480(v45, v65, 1, 0);
      v62[0] = v11;
      v66 = 257;
      v11 = sub_921880((unsigned int **)a1, v46, v61, (int)v62, 1, (__int64)v65, 0);
      *(_WORD *)(v11 + 2) = *(_WORD *)(v11 + 2) & 0xF003 | 0x20;
      v49 = *(unsigned int *)(a3 + 224);
      if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 228) )
      {
        sub_C8D5F0(a3 + 216, (const void *)(a3 + 232), v49 + 1, 8u, v47, v48);
        v49 = *(unsigned int *)(a3 + 224);
      }
      *(_QWORD *)(*(_QWORD *)(a3 + 216) + 8 * v49) = v11;
      ++*(_DWORD *)(a3 + 224);
      goto LABEL_32;
    }
    v31 = sub_AA5190(*(_QWORD *)(v3 - 96));
    if ( !v31 )
      BUG();
    v33 = HIBYTE(v32);
    LOBYTE(v34) = v32;
    v35 = *(_QWORD *)(v31 + 16);
    *(_QWORD *)(a1 + 56) = v31;
    HIBYTE(v34) = v33;
    *(_QWORD *)(a1 + 48) = v35;
    *(_WORD *)(a1 + 64) = v34;
    v36 = *(_QWORD *)sub_B46C60(v31 - 24);
    v65[0] = v36;
    if ( v36 && (sub_B96E90((__int64)v65, v36, 1), (v37 = v65[0]) != 0) )
    {
      v38 = *(unsigned int *)(a1 + 8);
      v39 = *(unsigned int **)a1;
      v40 = *(_DWORD *)(a1 + 8);
      v41 = (unsigned int *)(*(_QWORD *)a1 + 16 * v38);
      if ( *(unsigned int **)a1 != v41 )
      {
        while ( *v39 )
        {
          v39 += 4;
          if ( v41 == v39 )
            goto LABEL_57;
        }
        *((_QWORD *)v39 + 1) = v65[0];
LABEL_49:
        sub_B91220((__int64)v65, v37);
        goto LABEL_50;
      }
LABEL_57:
      v51 = *(unsigned int *)(a1 + 12);
      if ( v38 >= v51 )
      {
        v52 = v38 + 1;
        if ( v51 < v52 )
        {
          v59 = v65[0];
          sub_C8D5F0(a1, (const void *)(a1 + 16), v52, 0x10u, v65[0], a1 + 16);
          v37 = v59;
          v41 = (unsigned int *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *(_QWORD *)v41 = 0;
        *((_QWORD *)v41 + 1) = v37;
        v37 = v65[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v41 )
        {
          *v41 = 0;
          *((_QWORD *)v41 + 1) = v37;
          v40 = *(_DWORD *)(a1 + 8);
          v37 = v65[0];
        }
        *(_DWORD *)(a1 + 8) = v40 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v37 = v65[0];
    }
    if ( !v37 )
      goto LABEL_50;
    goto LABEL_49;
  }
LABEL_32:
  sub_BD84D0(v3, v11);
  return sub_B43D60((_QWORD *)v3);
}
