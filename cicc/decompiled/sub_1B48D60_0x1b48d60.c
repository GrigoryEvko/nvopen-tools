// Function: sub_1B48D60
// Address: 0x1b48d60
//
__int64 __fastcall sub_1B48D60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  char v7; // di
  unsigned int v8; // ebx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rcx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 **v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rsi
  __int64 v30; // rbx
  _QWORD *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r14
  __int64 v44; // r12
  int v45; // eax
  __int64 v46; // rax
  int v47; // edx
  __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  __int64 v54; // [rsp+28h] [rbp-58h]
  __int64 v55[2]; // [rsp+30h] [rbp-50h] BYREF
  char v56; // [rsp+40h] [rbp-40h]
  char v57; // [rsp+41h] [rbp-3Fh]

  v50 = sub_157F1C0(a2);
  v48 = *(_QWORD *)(v50 + 48);
  for ( i = v48; ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v5 = i - 24;
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v6 = 0x17FFFFFFE8LL;
    v7 = *(_BYTE *)(i - 1) & 0x40;
    v8 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v8 )
    {
      v9 = 24LL * *(unsigned int *)(i + 32) + 8;
      v10 = 0;
      do
      {
        v11 = v5 - 24LL * v8;
        if ( v7 )
          v11 = *(_QWORD *)(i - 32);
        if ( a2 == *(_QWORD *)(v11 + v9) )
        {
          v6 = 24 * v10;
          goto LABEL_11;
        }
        ++v10;
        v9 += 8;
      }
      while ( v8 != (_DWORD)v10 );
      v6 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v7 )
    {
      v12 = *(_QWORD *)(i - 32);
      if ( a1 != *(_QWORD *)(v12 + v6) )
        continue;
    }
    else
    {
      v12 = v5 - 24LL * v8;
      if ( a1 != *(_QWORD *)(v12 + v6) )
        continue;
    }
    if ( a3 )
    {
      v13 = *(_QWORD *)(v50 + 8);
      if ( v13 )
      {
        while ( 1 )
        {
          v53 = v13;
          v14 = sub_1648700(v13);
          v15 = v53;
          if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) <= 9u )
            break;
          v13 = *(_QWORD *)(v53 + 8);
          if ( !v13 )
            goto LABEL_33;
        }
      }
      else
      {
LABEL_33:
        v14 = sub_1648700(0);
        v15 = 0;
      }
      v16 = v14[5];
      if ( a2 == v16 )
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(v15 + 8);
          if ( !v21 )
            break;
          v54 = v21;
          v22 = sub_1648700(v21);
          if ( (unsigned __int8)(*((_BYTE *)v22 + 16) - 25) <= 9u )
            goto LABEL_31;
          v15 = v54;
        }
        v22 = sub_1648700(0);
LABEL_31:
        v16 = v22[5];
      }
      v17 = 0x17FFFFFFE8LL;
      if ( v8 )
      {
        v18 = 0;
        do
        {
          if ( v16 == *(_QWORD *)(v12 + 24LL * *(unsigned int *)(i + 32) + 8 * v18 + 8) )
          {
            v17 = 24 * v18;
            goto LABEL_24;
          }
          ++v18;
        }
        while ( v8 != (_DWORD)v18 );
        v17 = 0x17FFFFFFE8LL;
      }
LABEL_24:
      v19 = *(_QWORD *)(v12 + v17);
      if ( !v19 || a3 != v19 )
        continue;
    }
    return v5;
  }
  if ( a3 || (v5 = a1, *(_BYTE *)(a1 + 16) > 0x17u) && a2 == *(_QWORD *)(a1 + 40) )
  {
    v57 = 1;
    v56 = 3;
    v23 = v48 - 24;
    if ( !v48 )
      v23 = 0;
    v49 = v23;
    v55[0] = (__int64)"simplifycfg.merge";
    v24 = *(__int64 ***)a1;
    v25 = sub_1648B60(64);
    v5 = v25;
    if ( v25 )
    {
      sub_15F1EA0(v25, (__int64)v24, 53, 0, 0, v49);
      *(_DWORD *)(v5 + 56) = 2;
      sub_164B780(v5, v55);
      sub_1648880(v5, *(_DWORD *)(v5 + 56), 1);
    }
    v29 = a1;
    sub_1704F80(v5, a1, a2, v26, v27, v28);
    v55[0] = *(_QWORD *)(v50 + 8);
    sub_15CDD40(v55);
    v30 = v55[0];
    if ( v55[0] )
    {
      v31 = sub_1648700(v55[0]);
LABEL_55:
      v43 = v31[5];
      if ( a2 != v43 )
      {
        v44 = a3;
        if ( !a3 )
          v44 = sub_1599EF0(*(__int64 ***)a1);
        v45 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        if ( v45 == *(_DWORD *)(v5 + 56) )
        {
          sub_15F55D0(v5, v29, v32, v33, v34, v35);
          v45 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        }
        v46 = (v45 + 1) & 0xFFFFFFF;
        v47 = v46 | *(_DWORD *)(v5 + 20) & 0xF0000000;
        *(_DWORD *)(v5 + 20) = v47;
        if ( (v47 & 0x40000000) != 0 )
          v36 = *(_QWORD *)(v5 - 8);
        else
          v36 = v5 - 24 * v46;
        v37 = (__int64 *)(v36 + 24LL * (unsigned int)(v46 - 1));
        if ( *v37 )
        {
          v38 = v37[1];
          v39 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v39 = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = *(_QWORD *)(v38 + 16) & 3LL | v39;
        }
        *v37 = v44;
        if ( v44 )
        {
          v40 = *(_QWORD *)(v44 + 8);
          v37[1] = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v40 + 16) & 3LL;
          v37[2] = (v44 + 8) | v37[2] & 3;
          *(_QWORD *)(v44 + 8) = v37;
        }
        v41 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v42 = *(_QWORD *)(v5 - 8);
        else
          v42 = v5 - 24 * v41;
        *(_QWORD *)(v42 + 8LL * (unsigned int)(v41 - 1) + 24LL * *(unsigned int *)(v5 + 56) + 8) = v43;
      }
      while ( 1 )
      {
        v30 = *(_QWORD *)(v30 + 8);
        if ( !v30 )
          break;
        v31 = sub_1648700(v30);
        v29 = *((unsigned __int8 *)v31 + 16);
        v32 = (unsigned int)(v29 - 25);
        if ( (unsigned __int8)(v29 - 25) <= 9u )
          goto LABEL_55;
      }
    }
  }
  return v5;
}
