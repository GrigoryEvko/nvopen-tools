// Function: sub_F92230
// Address: 0xf92230
//
__int64 __fastcall sub_F92230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rax
  _QWORD *v24; // r10
  int v25; // eax
  int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // eax
  int v36; // eax
  unsigned int v37; // esi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-70h]
  _QWORD *v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  const char *v46; // [rsp+10h] [rbp-60h] BYREF
  char v47; // [rsp+30h] [rbp-40h]
  char v48; // [rsp+31h] [rbp-3Fh]

  v6 = sub_AA56F0(a2);
  v7 = *(_QWORD *)(v6 + 56);
  v8 = v6;
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    v9 = v7 - 24;
    if ( *(_BYTE *)(v7 - 24) != 84 )
      break;
    v10 = *(_QWORD *)(v7 - 32);
    v11 = 0x1FFFFFFFE0LL;
    v12 = *(_DWORD *)(v7 - 20) & 0x7FFFFFF;
    if ( v12 )
    {
      v13 = 0;
      do
      {
        if ( a2 == *(_QWORD *)(v10 + 32LL * *(unsigned int *)(v7 + 48) + 8 * v13) )
        {
          v11 = 32 * v13;
          goto LABEL_9;
        }
        ++v13;
      }
      while ( v12 != (_DWORD)v13 );
      if ( a1 != *(_QWORD *)(v10 + 0x1FFFFFFFE0LL) )
        goto LABEL_10;
    }
    else
    {
LABEL_9:
      if ( a1 != *(_QWORD *)(v10 + v11) )
        goto LABEL_10;
    }
    if ( !a3 )
      return v9;
    v14 = *(_QWORD *)(v8 + 16);
    if ( !v14 )
LABEL_66:
      BUG();
    while ( 1 )
    {
      v15 = *(_QWORD *)(v14 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v15 - 30) <= 0xAu )
        break;
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_66;
    }
    v16 = *(_QWORD *)(v15 + 40);
    if ( a2 == v16 )
    {
      do
      {
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
          BUG();
        v21 = *(_QWORD *)(v14 + 24);
      }
      while ( (unsigned __int8)(*(_BYTE *)v21 - 30) > 0xAu );
      v16 = *(_QWORD *)(v21 + 40);
    }
    v17 = 0x1FFFFFFFE0LL;
    if ( v12 )
    {
      v18 = 0;
      do
      {
        if ( v16 == *(_QWORD *)(v10 + 32LL * *(unsigned int *)(v7 + 48) + 8 * v18) )
        {
          v17 = 32 * v18;
          goto LABEL_21;
        }
        ++v18;
      }
      while ( v12 != (_DWORD)v18 );
      v17 = 0x1FFFFFFFE0LL;
    }
LABEL_21:
    v19 = *(_QWORD *)(v10 + v17);
    if ( a3 == v19 )
    {
      if ( v19 )
        return v9;
    }
LABEL_10:
    v7 = *(_QWORD *)(v7 + 8);
  }
  if ( a3 || (v9 = a1, *(_BYTE *)a1 > 0x1Cu) && a2 == *(_QWORD *)(a1 + 40) )
  {
    v22 = *(_QWORD *)(a1 + 8);
    v48 = 1;
    v46 = "simplifycfg.merge";
    v47 = 3;
    v23 = sub_BD2DA0(80);
    v9 = v23;
    if ( v23 )
    {
      v43 = (_QWORD *)v23;
      sub_B44260(v23, v22, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v9 + 72) = 2;
      sub_BD6B50((unsigned __int8 *)v9, &v46);
      sub_BD2A10(v9, *(_DWORD *)(v9 + 72), 1);
      v24 = v43;
    }
    else
    {
      v24 = 0;
    }
    sub_B44220(v24, *(_QWORD *)(v8 + 56), 1);
    v25 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
    if ( v25 == *(_DWORD *)(v9 + 72) )
    {
      sub_B48D90(v9);
      v25 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
    }
    v26 = (v25 + 1) & 0x7FFFFFF;
    v27 = v26 | *(_DWORD *)(v9 + 4) & 0xF8000000;
    v28 = *(_QWORD *)(v9 - 8) + 32LL * (unsigned int)(v26 - 1);
    *(_DWORD *)(v9 + 4) = v27;
    if ( *(_QWORD *)v28 )
    {
      v29 = *(_QWORD *)(v28 + 8);
      **(_QWORD **)(v28 + 16) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = *(_QWORD *)(v28 + 16);
    }
    *(_QWORD *)v28 = a1;
    v30 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v28 + 8) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = v28 + 8;
    *(_QWORD *)(v28 + 16) = a1 + 16;
    *(_QWORD *)(a1 + 16) = v28;
    *(_QWORD *)(*(_QWORD *)(v9 - 8) + 32LL * *(unsigned int *)(v9 + 72) + 8LL * ((*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) - 1)) = a2;
    v31 = *(_QWORD *)(v8 + 16);
    if ( v31 )
    {
      while ( 1 )
      {
        v32 = *(_QWORD *)(v31 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
          break;
        v31 = *(_QWORD *)(v31 + 8);
        if ( !v31 )
          return v9;
      }
LABEL_44:
      v33 = *(_QWORD *)(v32 + 40);
      if ( a2 != v33 )
      {
        v34 = a3;
        if ( !a3 )
        {
          v45 = v33;
          v41 = sub_ACADE0(*(__int64 ***)(a1 + 8));
          v33 = v45;
          v34 = v41;
        }
        v35 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
        if ( v35 == *(_DWORD *)(v9 + 72) )
        {
          v42 = v33;
          v44 = v34;
          sub_B48D90(v9);
          v33 = v42;
          v34 = v44;
          v35 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
        }
        v36 = (v35 + 1) & 0x7FFFFFF;
        v37 = v36 | *(_DWORD *)(v9 + 4) & 0xF8000000;
        v38 = *(_QWORD *)(v9 - 8) + 32LL * (unsigned int)(v36 - 1);
        *(_DWORD *)(v9 + 4) = v37;
        if ( *(_QWORD *)v38 )
        {
          v39 = *(_QWORD *)(v38 + 8);
          **(_QWORD **)(v38 + 16) = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
        }
        *(_QWORD *)v38 = v34;
        if ( v34 )
        {
          v40 = *(_QWORD *)(v34 + 16);
          *(_QWORD *)(v38 + 8) = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = v38 + 8;
          *(_QWORD *)(v38 + 16) = v34 + 16;
          *(_QWORD *)(v34 + 16) = v38;
        }
        *(_QWORD *)(*(_QWORD *)(v9 - 8)
                  + 32LL * *(unsigned int *)(v9 + 72)
                  + 8LL * ((*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) - 1)) = v33;
      }
      while ( 1 )
      {
        v31 = *(_QWORD *)(v31 + 8);
        if ( !v31 )
          break;
        v32 = *(_QWORD *)(v31 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v32 - 30) <= 0xAu )
          goto LABEL_44;
      }
    }
  }
  return v9;
}
