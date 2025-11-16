// Function: sub_28D2550
// Address: 0x28d2550
//
__int64 __fastcall sub_28D2550(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 *v17; // rcx
  __int64 *v18; // rsi
  _BYTE *v19; // rax
  _BYTE **v20; // rax
  int v21; // r14d
  _BYTE **v22; // rdx
  __int64 v23; // rdx
  _BYTE **v24; // rax
  _BYTE *v25; // rax
  unsigned int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // ecx
  __int64 v30; // rdx
  _BYTE *v31; // rdi
  int v32; // r11d
  __int64 v33; // r10
  int v34; // edi
  int v35; // ecx
  unsigned int v36; // esi
  int v37; // eax
  __int64 **v38; // rdx
  int v39; // eax
  int v40; // eax
  int v41; // r8d
  __int64 v42; // [rsp+18h] [rbp-A8h] BYREF
  _BYTE *v43; // [rsp+20h] [rbp-A0h] BYREF
  int v44; // [rsp+28h] [rbp-98h]
  _BYTE **v45; // [rsp+30h] [rbp-90h] BYREF
  _BYTE **v46; // [rsp+38h] [rbp-88h]
  __int64 v47; // [rsp+40h] [rbp-80h]
  __int64 v48; // [rsp+48h] [rbp-78h]
  _QWORD v49[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v50; // [rsp+70h] [rbp-50h] BYREF
  __int64 v51; // [rsp+78h] [rbp-48h]
  __int64 v52; // [rsp+80h] [rbp-40h]
  __int64 v53; // [rsp+88h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 2008);
  v5 = *(_QWORD *)(a1 + 1992);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      v10 = *((_DWORD *)v8 + 2);
      if ( v10 )
      {
        LOBYTE(a2) = v10 != 2;
        return (unsigned int)a2;
      }
    }
    else
    {
      v40 = 1;
      while ( v9 != -4096 )
      {
        v41 = v40 + 1;
        v7 = v6 & (v40 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v40 = v41;
      }
    }
  }
  v50 = (__int64 *)a2;
  v11 = sub_28CBE90(a1 + 352, (__int64 *)&v50);
  if ( !v11 || !*((_DWORD *)v11 + 2) )
    sub_28D1FB0(a1 + 248, a2);
  v50 = (__int64 *)a2;
  v12 = 0;
  v13 = sub_28CBE90(a1 + 1248, (__int64 *)&v50);
  if ( v13 )
    v12 = 96LL * *((unsigned int *)v13 + 2);
  v14 = *(_QWORD *)(a1 + 464) + v12;
  v15 = *(unsigned int *)(v14 + 20);
  if ( *(_DWORD *)(v14 + 20) - *(_DWORD *)(v14 + 24) == 1 )
  {
    v50 = (__int64 *)a2;
    LODWORD(v51) = 1;
    if ( (unsigned __int8)sub_28C76B0(a1 + 1984, (__int64 *)&v50, &v45) )
    {
LABEL_47:
      LODWORD(a2) = 1;
      return (unsigned int)a2;
    }
    v36 = *(_DWORD *)(a1 + 2008);
    v37 = *(_DWORD *)(a1 + 2000);
    v38 = (__int64 **)v45;
    ++*(_QWORD *)(a1 + 1984);
    v39 = v37 + 1;
    v49[0] = v38;
    if ( 4 * v39 >= 3 * v36 )
    {
      v36 *= 2;
    }
    else if ( v36 - *(_DWORD *)(a1 + 2004) - v39 > v36 >> 3 )
    {
LABEL_51:
      *(_DWORD *)(a1 + 2000) = v39;
      if ( *v38 != (__int64 *)-4096LL )
        --*(_DWORD *)(a1 + 2004);
      *v38 = v50;
      *((_DWORD *)v38 + 2) = v51;
      goto LABEL_47;
    }
    sub_28C9CC0(a1 + 1984, v36);
    sub_28C76B0(a1 + 1984, (__int64 *)&v50, v49);
    v38 = (__int64 **)v49[0];
    v39 = *(_DWORD *)(a1 + 2000) + 1;
    goto LABEL_51;
  }
  if ( !*(_BYTE *)(v14 + 28) )
    v15 = *(unsigned int *)(v14 + 16);
  v16 = *(_QWORD *)(v14 + 8) + 8 * v15;
  v50 = *(__int64 **)(v14 + 8);
  v51 = v16;
  sub_254BBF0((__int64)&v50);
  v52 = v14;
  v17 = v50;
  v18 = (__int64 *)v51;
  v53 = *(_QWORD *)v14;
  if ( v50 == (__int64 *)v16 )
  {
LABEL_30:
    v20 = *(_BYTE ***)(v14 + 8);
    LODWORD(a2) = 1;
    v21 = 1;
    if ( *(_BYTE *)(v14 + 28) )
    {
LABEL_17:
      v22 = &v20[*(unsigned int *)(v14 + 20)];
      goto LABEL_18;
    }
  }
  else
  {
    while ( 1 )
    {
      if ( *(_BYTE *)*v17 != 84 )
      {
        v19 = (_BYTE *)sub_28C8570(*v17);
        if ( !v19 || *v19 != 84 )
          break;
      }
      do
        ++v17;
      while ( v17 != v18 && (unsigned __int64)*v17 >= 0xFFFFFFFFFFFFFFFELL );
      if ( v17 == (__int64 *)v16 )
        goto LABEL_30;
    }
    LODWORD(a2) = 0;
    v20 = *(_BYTE ***)(v14 + 8);
    v21 = 2;
    if ( *(_BYTE *)(v14 + 28) )
      goto LABEL_17;
  }
  v22 = &v20[*(unsigned int *)(v14 + 16)];
LABEL_18:
  v45 = v20;
  v46 = v22;
  sub_254BBF0((__int64)&v45);
  v47 = v14;
  v48 = *(_QWORD *)v14;
  if ( *(_BYTE *)(v14 + 28) )
    v23 = *(unsigned int *)(v14 + 20);
  else
    v23 = *(unsigned int *)(v14 + 16);
  v49[0] = *(_QWORD *)(v14 + 8) + 8 * v23;
  v49[1] = v49[0];
  sub_254BBF0((__int64)v49);
  v49[2] = v14;
  v49[3] = *(_QWORD *)v14;
  v24 = v45;
  if ( v45 != (_BYTE **)v49[0] )
  {
    while ( 1 )
    {
      v25 = *v24;
      if ( *v25 == 84 )
      {
        v27 = *(_DWORD *)(a1 + 2008);
        v43 = v25;
        v44 = v21;
        if ( !v27 )
        {
          ++*(_QWORD *)(a1 + 1984);
          v42 = 0;
          goto LABEL_59;
        }
        v28 = *(_QWORD *)(a1 + 1992);
        v29 = (v27 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v30 = v28 + 16LL * v29;
        v31 = *(_BYTE **)v30;
        if ( v25 != *(_BYTE **)v30 )
        {
          v32 = 1;
          v33 = 0;
          while ( v31 != (_BYTE *)-4096LL )
          {
            if ( v31 == (_BYTE *)-8192LL && !v33 )
              v33 = v30;
            v29 = (v27 - 1) & (v32 + v29);
            v30 = v28 + 16LL * v29;
            v31 = *(_BYTE **)v30;
            if ( v25 == *(_BYTE **)v30 )
              goto LABEL_22;
            ++v32;
          }
          v34 = *(_DWORD *)(a1 + 2000);
          if ( v33 )
            v30 = v33;
          ++*(_QWORD *)(a1 + 1984);
          v35 = v34 + 1;
          v42 = v30;
          if ( 4 * (v34 + 1) < 3 * v27 )
          {
            if ( v27 - *(_DWORD *)(a1 + 2004) - v35 > v27 >> 3 )
            {
LABEL_43:
              *(_DWORD *)(a1 + 2000) = v35;
              if ( *(_QWORD *)v30 != -4096 )
                --*(_DWORD *)(a1 + 2004);
              *(_QWORD *)v30 = v25;
              *(_DWORD *)(v30 + 8) = v44;
              goto LABEL_22;
            }
LABEL_60:
            sub_28C9CC0(a1 + 1984, v27);
            sub_28C76B0(a1 + 1984, (__int64 *)&v43, &v42);
            v25 = v43;
            v30 = v42;
            v35 = *(_DWORD *)(a1 + 2000) + 1;
            goto LABEL_43;
          }
LABEL_59:
          v27 *= 2;
          goto LABEL_60;
        }
      }
LABEL_22:
      v24 = v45 + 1;
      v45 = v24;
      if ( v24 == v46 )
      {
LABEL_25:
        if ( (_BYTE **)v49[0] == v24 )
          return (unsigned int)a2;
      }
      else
      {
        while ( (unsigned __int64)(*v24 + 2) <= 1 )
        {
          v45 = ++v24;
          if ( v46 == v24 )
            goto LABEL_25;
        }
        v24 = v45;
        if ( (_BYTE **)v49[0] == v45 )
          return (unsigned int)a2;
      }
    }
  }
  return (unsigned int)a2;
}
