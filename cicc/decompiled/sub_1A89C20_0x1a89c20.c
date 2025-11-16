// Function: sub_1A89C20
// Address: 0x1a89c20
//
void __fastcall sub_1A89C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rax
  __int64 *v7; // r14
  unsigned __int64 *v8; // r15
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r9
  unsigned __int64 v12; // rax
  unsigned __int64 *v13; // rdi
  unsigned int v14; // r10d
  unsigned __int64 *v15; // rdx
  unsigned __int64 *v16; // rcx
  __int64 v17; // rax
  unsigned __int64 *v18; // r14
  _BYTE *v19; // rdi
  unsigned __int64 *v20; // rax
  signed __int64 v21; // r15
  unsigned __int64 *v22; // rdx
  unsigned __int64 *v23; // rcx
  unsigned __int64 v24; // rdx
  unsigned __int64 *v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 *v30; // r14
  __int64 *v31; // rbx
  _QWORD *v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // rax
  __int64 *v35; // rax
  char v36; // dl
  __int64 v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rsi
  _QWORD *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 *v43; // rsi
  unsigned int v44; // edi
  __int64 *v45; // rcx
  __int64 v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // [rsp+10h] [rbp-90h]
  __int64 *v49; // [rsp+18h] [rbp-88h]
  _BYTE *v50; // [rsp+20h] [rbp-80h] BYREF
  __int64 v51; // [rsp+28h] [rbp-78h]
  _BYTE v52[112]; // [rsp+30h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a1 + 112);
  v7 = *(__int64 **)(v6 + 32);
  v49 = *(__int64 **)(v6 + 40);
  if ( v49 != v7 )
  {
    v8 = *(unsigned __int64 **)(a1 + 16);
    v9 = *(unsigned __int64 **)(a1 + 8);
    while ( 1 )
    {
LABEL_5:
      v10 = v8;
      v12 = sub_157EBA0(*v7);
      v11 = v9;
      if ( v8 != v9 )
        goto LABEL_3;
      v13 = &v8[*(unsigned int *)(a1 + 28)];
      v14 = *(_DWORD *)(a1 + 28);
      if ( v8 != v13 )
      {
        v15 = v8;
        v16 = 0;
        while ( v12 != *v15 )
        {
          if ( *v15 == -2 )
            v16 = v15;
          if ( v13 == ++v15 )
          {
            if ( !v16 )
              goto LABEL_84;
            *v16 = v12;
            v8 = *(unsigned __int64 **)(a1 + 16);
            ++v7;
            v9 = *(unsigned __int64 **)(a1 + 8);
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            v10 = v8;
            v11 = v9;
            if ( v49 != v7 )
              goto LABEL_5;
            goto LABEL_14;
          }
        }
        goto LABEL_4;
      }
LABEL_84:
      if ( v14 < *(_DWORD *)(a1 + 24) )
      {
        *(_DWORD *)(a1 + 28) = v14 + 1;
        *v13 = v12;
        v9 = *(unsigned __int64 **)(a1 + 8);
        v8 = *(unsigned __int64 **)(a1 + 16);
        ++*(_QWORD *)a1;
        v11 = v9;
        v10 = v8;
      }
      else
      {
LABEL_3:
        sub_16CCBA0(a1, v12);
        v8 = *(unsigned __int64 **)(a1 + 16);
        v9 = *(unsigned __int64 **)(a1 + 8);
        v10 = v8;
        v11 = v9;
      }
LABEL_4:
      if ( v49 == ++v7 )
        goto LABEL_14;
    }
  }
  v10 = *(unsigned __int64 **)(a1 + 16);
  v11 = *(unsigned __int64 **)(a1 + 8);
LABEL_14:
  if ( v11 == v10 )
    v17 = *(unsigned int *)(a1 + 28);
  else
    v17 = *(unsigned int *)(a1 + 24);
  v18 = &v10[v17];
  while ( 1 )
  {
    if ( v10 == v18 )
      goto LABEL_20;
    if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v10;
  }
  v50 = v52;
  v51 = 0x800000000LL;
  if ( v10 == v18 )
  {
LABEL_20:
    v19 = v52;
  }
  else
  {
    v20 = v10;
    v21 = 0;
    while ( 1 )
    {
      v22 = v20 + 1;
      if ( v20 + 1 == v18 )
        break;
      while ( 1 )
      {
        v20 = v22;
        if ( *v22 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v18 == ++v22 )
          goto LABEL_29;
      }
      ++v21;
      if ( v22 == v18 )
        goto LABEL_30;
    }
LABEL_29:
    ++v21;
LABEL_30:
    v23 = (unsigned __int64 *)v52;
    if ( v21 > 8 )
    {
      sub_16CD150((__int64)&v50, v52, v21, 8, a5, (int)v11);
      v23 = (unsigned __int64 *)&v50[8 * (unsigned int)v51];
    }
    v24 = *v10;
    do
    {
      v25 = v10 + 1;
      *v23++ = v24;
      if ( v10 + 1 == v18 )
        break;
      while ( 1 )
      {
        v24 = *v25;
        v10 = v25;
        if ( *v25 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v18 == ++v25 )
          goto LABEL_36;
      }
    }
    while ( v25 != v18 );
LABEL_36:
    v19 = v50;
    LODWORD(v51) = v21 + v51;
    v26 = v51;
    if ( (_DWORD)v51 )
    {
      while ( 1 )
      {
        v27 = v26--;
        v28 = *(_QWORD *)&v19[8 * v27 - 8];
        LODWORD(v51) = v26;
        v29 = 3LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v28 + 23) & 0x40) != 0 )
        {
          v30 = *(__int64 **)(v28 - 8);
          v31 = &v30[v29];
        }
        else
        {
          v31 = (__int64 *)v28;
          v30 = (__int64 *)(v28 - v29 * 8);
        }
        if ( v31 != v30 )
          break;
LABEL_63:
        if ( !v26 )
          goto LABEL_21;
      }
      while ( 1 )
      {
        v37 = *v30;
        if ( *(_BYTE *)(*v30 + 16) <= 0x17u )
          goto LABEL_47;
        v38 = *(_QWORD *)(a1 + 112);
        v39 = *(_QWORD *)(v37 + 40);
        v40 = *(_QWORD **)(v38 + 72);
        v33 = *(_QWORD **)(v38 + 64);
        if ( v40 == v33 )
        {
          v32 = &v33[*(unsigned int *)(v38 + 84)];
          if ( v33 == v32 )
          {
            v47 = *(_QWORD **)(v38 + 64);
          }
          else
          {
            do
            {
              if ( v39 == *v33 )
                break;
              ++v33;
            }
            while ( v32 != v33 );
            v47 = v32;
          }
        }
        else
        {
          v48 = *(_QWORD *)(a1 + 112);
          v32 = &v40[*(unsigned int *)(v38 + 80)];
          v33 = sub_16CC9F0(v38 + 56, v39);
          if ( v39 == *v33 )
          {
            v41 = *(_QWORD *)(v48 + 72);
            if ( v41 == *(_QWORD *)(v48 + 64) )
              v42 = *(unsigned int *)(v48 + 84);
            else
              v42 = *(unsigned int *)(v48 + 80);
            v47 = (_QWORD *)(v41 + 8 * v42);
          }
          else
          {
            v34 = *(_QWORD *)(v48 + 72);
            if ( v34 != *(_QWORD *)(v48 + 64) )
            {
              v33 = (_QWORD *)(v34 + 8LL * *(unsigned int *)(v48 + 80));
              goto LABEL_44;
            }
            v33 = (_QWORD *)(v34 + 8LL * *(unsigned int *)(v48 + 84));
            v47 = v33;
          }
        }
        while ( v47 != v33 && *v33 >= 0xFFFFFFFFFFFFFFFELL )
          ++v33;
LABEL_44:
        if ( v32 != v33 )
        {
          v35 = *(__int64 **)(a1 + 8);
          if ( *(__int64 **)(a1 + 16) == v35 )
          {
            v43 = &v35[*(unsigned int *)(a1 + 28)];
            v44 = *(_DWORD *)(a1 + 28);
            if ( v35 != v43 )
            {
              v45 = 0;
              while ( v37 != *v35 )
              {
                if ( *v35 == -2 )
                  v45 = v35;
                if ( v43 == ++v35 )
                {
                  if ( !v45 )
                    goto LABEL_81;
                  *v45 = v37;
                  --*(_DWORD *)(a1 + 32);
                  ++*(_QWORD *)a1;
                  goto LABEL_77;
                }
              }
              goto LABEL_47;
            }
LABEL_81:
            if ( v44 < *(_DWORD *)(a1 + 24) )
            {
              *(_DWORD *)(a1 + 28) = v44 + 1;
              *v43 = v37;
              v46 = (unsigned int)v51;
              ++*(_QWORD *)a1;
              if ( (unsigned int)v46 >= HIDWORD(v51) )
              {
LABEL_83:
                sub_16CD150((__int64)&v50, v52, 0, 8, v28, (int)v11);
                v46 = (unsigned int)v51;
              }
LABEL_78:
              *(_QWORD *)&v50[8 * v46] = v37;
              LODWORD(v51) = v51 + 1;
              goto LABEL_47;
            }
          }
          sub_16CCBA0(a1, v37);
          if ( v36 )
          {
LABEL_77:
            v46 = (unsigned int)v51;
            if ( (unsigned int)v51 >= HIDWORD(v51) )
              goto LABEL_83;
            goto LABEL_78;
          }
        }
LABEL_47:
        v30 += 3;
        if ( v31 == v30 )
        {
          v26 = v51;
          v19 = v50;
          goto LABEL_63;
        }
      }
    }
  }
LABEL_21:
  if ( v19 != v52 )
    _libc_free((unsigned __int64)v19);
}
