// Function: sub_F8A290
// Address: 0xf8a290
//
unsigned __int8 *__fastcall sub_F8A290(__int64 a1, __int64 a2, _BYTE *a3, char a4)
{
  _BYTE *v7; // rax
  __int64 v8; // r14
  unsigned __int8 *v9; // r15
  __int64 v11; // rax
  _QWORD *v12; // rdx
  _QWORD *v13; // rsi
  int v14; // ecx
  _QWORD *v15; // rdx
  _BYTE *v16; // rax
  char v17; // al
  __int64 v18; // rax
  __int16 v19; // ax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  int v23; // edi
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // r8
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // rbx
  __int64 v39; // rax
  int v40; // eax
  int v41; // eax
  int v42; // r9d
  _QWORD *v43; // [rsp+10h] [rbp-B0h]
  int v44; // [rsp+18h] [rbp-A8h]
  unsigned int v45; // [rsp+1Ch] [rbp-A4h]
  const char *v46; // [rsp+20h] [rbp-A0h] BYREF
  char v47; // [rsp+40h] [rbp-80h]
  char v48; // [rsp+41h] [rbp-7Fh]
  __int64 v49; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v50[3]; // [rsp+58h] [rbp-68h] BYREF
  __int64 v51; // [rsp+70h] [rbp-50h]
  __int16 v52; // [rsp+78h] [rbp-48h]
  _QWORD v53[8]; // [rsp+80h] [rbp-40h] BYREF

  v7 = (_BYTE *)sub_F894B0(a1, a2);
  v8 = (__int64)v7;
  v45 = (2 * a4) & 4;
  if ( *a3 <= 0x15u && *v7 <= 0x15u )
  {
    LOWORD(v51) = 257;
    return (unsigned __int8 *)sub_F7CA10((__int64 *)(a1 + 520), (__int64)a3, (__int64)v7, (__int64)&v49, (2 * a4) & 4);
  }
  v11 = *(_QWORD *)(a1 + 568);
  v12 = *(_QWORD **)(a1 + 576);
  v13 = *(_QWORD **)(v11 + 56);
  if ( v13 != v12 )
  {
    v14 = 6;
    v15 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
    while ( 1 )
    {
      if ( !v15 )
        BUG();
      v17 = *((_BYTE *)v15 - 24);
      if ( v17 != 85 )
        break;
      v18 = *(v15 - 7);
      if ( !v18 || *(_BYTE *)v18 || *(_QWORD *)(v18 + 24) != v15[7] || (*(_BYTE *)(v18 + 33) & 0x20) == 0 )
        goto LABEL_10;
      v14 += (unsigned int)(*(_DWORD *)(v18 + 36) - 68) < 4;
      if ( v15 == v13 )
      {
LABEL_19:
        v11 = *(_QWORD *)(a1 + 568);
        goto LABEL_20;
      }
LABEL_11:
      v15 = (_QWORD *)(*v15 & 0xFFFFFFFFFFFFFFF8LL);
      if ( !--v14 )
        goto LABEL_19;
    }
    if ( v17 == 63 )
    {
      v9 = (unsigned __int8 *)(v15 - 3);
      v16 = (_BYTE *)v15[-4 * (*((_DWORD *)v15 - 5) & 0x7FFFFFF) - 3];
      if ( a3 == v16 )
      {
        if ( v16 )
        {
          v38 = v15[6];
          v44 = v14;
          v43 = v15;
          v39 = sub_BCB2B0(*(_QWORD **)(a1 + 592));
          v15 = v43;
          v14 = v44;
          if ( v38 == v39 && v8 == *(_QWORD *)&v9[32 * (1LL - (*((_DWORD *)v43 - 5) & 0x7FFFFFF))] )
          {
            sub_F83EF0(a1, v9);
            v40 = sub_B4DE20((__int64)v9);
            sub_B4DDE0((__int64)v9, v40 & v45);
            return v9;
          }
        }
      }
    }
LABEL_10:
    if ( v15 == v13 )
      goto LABEL_19;
    goto LABEL_11;
  }
LABEL_20:
  v50[2] = v11;
  v49 = a1 + 520;
  v50[0] = 0;
  v50[1] = 0;
  if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
    sub_BD73F0((__int64)v50);
  v19 = *(_WORD *)(a1 + 584);
  v51 = *(_QWORD *)(a1 + 576);
  v52 = v19;
  sub_B33910(v53, (__int64 *)(a1 + 520));
  v22 = *(unsigned int *)(a1 + 792);
  v53[1] = a1;
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 796) )
  {
    sub_C8D5F0(a1 + 784, (const void *)(a1 + 800), v22 + 1, 8u, v20, v21);
    v22 = *(unsigned int *)(a1 + 792);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 784) + 8 * v22) = &v49;
  ++*(_DWORD *)(a1 + 792);
  while ( 1 )
  {
    v34 = *(_QWORD *)(a1 + 568);
    v35 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
    v36 = *(_QWORD *)(v35 + 8);
    v37 = *(_DWORD *)(v35 + 24);
    if ( !v37 )
      break;
    v23 = v37 - 1;
    v24 = (v37 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v25 = (__int64 *)(v36 + 16 * v24);
    v26 = *v25;
    if ( v34 != *v25 )
    {
      v41 = 1;
      while ( v26 != -4096 )
      {
        v42 = v41 + 1;
        v24 = v23 & (unsigned int)(v41 + v24);
        v25 = (__int64 *)(v36 + 16LL * (unsigned int)v24);
        v26 = *v25;
        if ( v34 == *v25 )
          goto LABEL_27;
        v41 = v42;
      }
      break;
    }
LABEL_27:
    v27 = v25[1];
    if ( !v27 )
      break;
    if ( !(unsigned __int8)sub_D48480(v25[1], (__int64)a3, v24, v34) )
      break;
    if ( !(unsigned __int8)sub_D48480(v27, v8, v28, v29) )
      break;
    v30 = sub_D4B130(v27);
    if ( !v30 )
      break;
    v31 = *(_QWORD *)(v30 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v31 == v30 + 48 )
    {
      v33 = 0;
    }
    else
    {
      if ( !v31 )
        BUG();
      v32 = *(unsigned __int8 *)(v31 - 24);
      v33 = v31 - 24;
      if ( (unsigned int)(v32 - 30) >= 0xB )
        v33 = 0;
    }
    sub_D5F1F0(a1 + 520, v33);
  }
  v48 = 1;
  v46 = "scevgep";
  v47 = 3;
  v9 = (unsigned __int8 *)sub_F7CA10((__int64 *)(a1 + 520), (__int64)a3, v8, (__int64)&v46, v45);
  sub_F80960((__int64)&v49);
  return v9;
}
