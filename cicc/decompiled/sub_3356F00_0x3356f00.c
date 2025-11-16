// Function: sub_3356F00
// Address: 0x3356f00
//
__int64 __fastcall sub_3356F00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(__int64 a1, __int64 a2),
        __int64 a5,
        __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r14
  unsigned __int64 v10; // r12
  int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // rdi
  void (__fastcall *v14)(__int64, __int64); // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 result; // rax
  char v19; // al
  __int64 v20; // rdi
  unsigned int *v21; // rax
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // r12
  char v25; // di
  __int64 v26; // r9
  int v27; // esi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r10
  __int64 *v31; // rax
  __int64 v32; // rax
  unsigned int v33; // esi
  unsigned int v34; // edx
  __int64 *v35; // rax
  int v36; // ecx
  unsigned int v37; // r10d
  int v38; // r14d
  __int64 v39; // r9
  int v40; // esi
  unsigned int v41; // ecx
  __int64 v42; // r11
  __int64 v43; // r9
  int v44; // esi
  unsigned int v45; // ecx
  __int64 v46; // r11
  __int64 *v47; // rdx
  int v48; // edi
  int v49; // esi
  int v50; // esi
  int v51; // edi
  unsigned __int64 v52; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v53[14]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a2;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = v8 + 16LL * *(unsigned int *)(a2 + 48);
  while ( v9 != v8 )
  {
    v10 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    v52 = v10;
    v11 = *(_DWORD *)(v10 + 220) - 1;
    *(_DWORD *)(v10 + 220) = v11;
    if ( *(_BYTE *)(a1 + 632) )
    {
      if ( (*(_BYTE *)(v7 + 254) & 2) == 0 )
        sub_2F8F770(v7, (_QWORD *)a2, a3, (__int64)a4, a5, a6);
      a2 = (unsigned int)(*(_DWORD *)(v7 + 244) + *(_DWORD *)(v8 + 12));
      sub_2F8F8C0(v10, (_QWORD *)a2, a3, (__int64)a4, a5, a6);
      v10 = v52;
      v11 = *(_DWORD *)(v52 + 220);
    }
    if ( v11 || v10 == a1 + 72 )
      goto LABEL_21;
    *(_BYTE *)(v10 + 249) |= 2u;
    a6 = v10;
    if ( (*(_BYTE *)(v10 + 254) & 2) == 0 )
    {
      sub_2F8F770(v10, (_QWORD *)a2, a3, (__int64)a4, a5, v10);
      a6 = v52;
    }
    v12 = *(_DWORD *)(v10 + 244);
    if ( v12 < *(_DWORD *)(a1 + 684) )
      *(_DWORD *)(a1 + 684) = v12;
    v13 = *(_QWORD *)(a1 + 640);
    if ( !byte_5038F08 && *(_BYTE *)(v13 + 12) )
    {
      a2 = a6;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v13 + 80LL))(v13, a6) )
      {
        a3 = v52;
        v19 = *(_BYTE *)(v52 + 249);
        if ( (v19 & 1) == 0 )
        {
          *(_BYTE *)(v52 + 249) = v19 | 1;
          a2 = *(_QWORD *)(a1 + 656);
          if ( a2 == *(_QWORD *)(a1 + 664) )
          {
            sub_2ECAD30(a1 + 648, (_BYTE *)a2, &v52);
          }
          else
          {
            if ( a2 )
            {
              *(_QWORD *)a2 = a3;
              a2 = *(_QWORD *)(a1 + 656);
            }
            a2 += 8;
            *(_QWORD *)(a1 + 656) = a2;
          }
        }
        goto LABEL_21;
      }
      v13 = *(_QWORD *)(a1 + 640);
      a6 = v52;
    }
    a4 = sub_33549A0;
    v14 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v13 + 88LL);
    if ( v14 == sub_33549A0 )
    {
      *(_QWORD *)v53 = a6;
      v15 = *(_DWORD *)(v13 + 40) + 1;
      *(_DWORD *)(v13 + 40) = v15;
      *(_DWORD *)(a6 + 204) = v15;
      a2 = *(_QWORD *)(v13 + 24);
      if ( a2 == *(_QWORD *)(v13 + 32) )
      {
        sub_2ECAD30(v13 + 16, (_BYTE *)a2, v53);
      }
      else
      {
        if ( a2 )
        {
          *(_QWORD *)a2 = a6;
          a2 = *(_QWORD *)(v13 + 24);
        }
        a2 += 8;
        *(_QWORD *)(v13 + 24) = a2;
      }
    }
    else
    {
      a2 = a6;
      v14(v13, a6);
    }
LABEL_21:
    if ( (*(_QWORD *)v8 & 6) == 0 )
    {
      a3 = *(unsigned int *)(v8 + 8);
      if ( (_DWORD)a3 )
      {
        a4 = *(void (__fastcall **)(__int64, __int64))(a1 + 696);
        *((_QWORD *)a4 + a3) = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
        v16 = *(_QWORD *)(a1 + 704);
        a3 = *(unsigned int *)(v8 + 8);
        if ( !*(_QWORD *)(v16 + 8 * a3) )
        {
          ++*(_DWORD *)(a1 + 692);
          a3 = *(unsigned int *)(v8 + 8);
          *(_QWORD *)(v16 + 8 * a3) = v7;
        }
      }
    }
    v8 += 16;
  }
  v17 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 16LL);
  result = *(_QWORD *)(a1 + 696);
  if ( !*(_QWORD *)(result + 8 * v17) )
  {
    v20 = *(_QWORD *)v7;
    while ( 1 )
    {
      if ( !v20 )
        return result;
      v22 = *(_DWORD *)(v20 + 24);
      if ( v22 < 0 )
      {
        v23 = *(_QWORD *)(a1 + 16);
        if ( *(_DWORD *)(v23 + 68) == ~v22 )
          break;
      }
      result = *(unsigned int *)(v20 + 64);
      if ( (_DWORD)result )
      {
        v21 = (unsigned int *)(*(_QWORD *)(v20 + 40) + 40LL * (unsigned int)(result - 1));
        v20 = *(_QWORD *)v21;
        result = *(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2];
        if ( *(_WORD *)result == 262 )
          continue;
      }
      return result;
    }
    LODWORD(v52) = 0;
    v53[0] = 0;
    v24 = *(_QWORD *)(a1 + 48) + ((__int64)*(int *)(sub_33513B0(v20, (unsigned int *)&v52, v53, v23) + 36) << 8);
    v25 = *(_BYTE *)(a1 + 1216) & 1;
    if ( v25 )
    {
      v26 = a1 + 1224;
      v27 = 15;
    }
    else
    {
      v33 = *(_DWORD *)(a1 + 1232);
      v26 = *(_QWORD *)(a1 + 1224);
      if ( !v33 )
      {
        v34 = *(_DWORD *)(a1 + 1216);
        ++*(_QWORD *)(a1 + 1208);
        v35 = 0;
        v36 = (v34 >> 1) + 1;
        goto LABEL_50;
      }
      v27 = v33 - 1;
    }
    v28 = v27 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v29 = (__int64 *)(v26 + 16LL * v28);
    v30 = *v29;
    if ( v24 == *v29 )
    {
LABEL_43:
      v31 = v29 + 1;
LABEL_44:
      *v31 = v7;
      v32 = *(_QWORD *)(a1 + 696);
      ++*(_DWORD *)(a1 + 692);
      *(_QWORD *)(v32 + 8 * v17) = v24;
      result = *(_QWORD *)(a1 + 704);
      *(_QWORD *)(result + 8 * v17) = v7;
      return result;
    }
    v35 = 0;
    v38 = 1;
    while ( v30 != -4096 )
    {
      if ( v30 == -8192 && !v35 )
        v35 = v29;
      v28 = v27 & (v38 + v28);
      v29 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v29;
      if ( v24 == *v29 )
        goto LABEL_43;
      ++v38;
    }
    v37 = 48;
    v33 = 16;
    if ( !v35 )
      v35 = v29;
    v34 = *(_DWORD *)(a1 + 1216);
    ++*(_QWORD *)(a1 + 1208);
    v36 = (v34 >> 1) + 1;
    if ( v25 )
    {
LABEL_51:
      if ( 4 * v36 < v37 )
      {
        if ( v33 - *(_DWORD *)(a1 + 1220) - v36 > v33 >> 3 )
        {
LABEL_53:
          *(_DWORD *)(a1 + 1216) = (2 * (v34 >> 1) + 2) | v34 & 1;
          if ( *v35 != -4096 )
            --*(_DWORD *)(a1 + 1220);
          *v35 = v24;
          v31 = v35 + 1;
          *v31 = 0;
          goto LABEL_44;
        }
        sub_3356AC0(a1 + 1208, v33);
        if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
        {
          v43 = a1 + 1224;
          v44 = 15;
          goto LABEL_68;
        }
        v50 = *(_DWORD *)(a1 + 1232);
        v43 = *(_QWORD *)(a1 + 1224);
        if ( v50 )
        {
          v44 = v50 - 1;
LABEL_68:
          v45 = v44 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v35 = (__int64 *)(v43 + 16LL * v45);
          v46 = *v35;
          if ( v24 != *v35 )
          {
            v47 = (__int64 *)(v43 + 16LL * (v44 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4))));
            v48 = 1;
            v35 = 0;
            while ( v46 != -4096 )
            {
              if ( v46 == -8192 && !v35 )
                v35 = v47;
              v45 = v44 & (v48 + v45);
              v47 = (__int64 *)(v43 + 16LL * v45);
              v46 = *v47;
              if ( v24 == *v47 )
                goto LABEL_72;
              ++v48;
            }
LABEL_71:
            if ( !v35 )
LABEL_72:
              v35 = v47;
            goto LABEL_65;
          }
          goto LABEL_65;
        }
LABEL_94:
        *(_DWORD *)(a1 + 1216) = (2 * (*(_DWORD *)(a1 + 1216) >> 1) + 2) | *(_DWORD *)(a1 + 1216) & 1;
        BUG();
      }
      sub_3356AC0(a1 + 1208, 2 * v33);
      if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
      {
        v39 = a1 + 1224;
        v40 = 15;
      }
      else
      {
        v49 = *(_DWORD *)(a1 + 1232);
        v39 = *(_QWORD *)(a1 + 1224);
        if ( !v49 )
          goto LABEL_94;
        v40 = v49 - 1;
      }
      v41 = v40 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v35 = (__int64 *)(v39 + 16LL * v41);
      v42 = *v35;
      if ( v24 != *v35 )
      {
        v47 = (__int64 *)(v39 + 16LL * (v40 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4))));
        v51 = 1;
        v35 = 0;
        while ( v42 != -4096 )
        {
          if ( !v35 && v42 == -8192 )
            v35 = v47;
          v41 = v40 & (v51 + v41);
          v47 = (__int64 *)(v39 + 16LL * v41);
          v42 = *v47;
          if ( v24 == *v47 )
            goto LABEL_72;
          ++v51;
        }
        goto LABEL_71;
      }
LABEL_65:
      v34 = *(_DWORD *)(a1 + 1216);
      goto LABEL_53;
    }
    v33 = *(_DWORD *)(a1 + 1232);
LABEL_50:
    v37 = 3 * v33;
    goto LABEL_51;
  }
  return result;
}
