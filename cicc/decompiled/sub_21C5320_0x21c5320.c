// Function: sub_21C5320
// Address: 0x21c5320
//
char __fastcall sub_21C5320(__int64 a1, __int64 a2, int a3)
{
  int v3; // edx
  char result; // al
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 *v7; // rbx
  char v8; // cl
  __int64 v9; // rbx
  __int64 v10; // r12
  void *v11; // rbx
  __int64 *v12; // rdi
  __int64 v13; // rdi
  unsigned int v14; // ebx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // ebx
  unsigned __int64 v19; // rdx
  int v20; // eax
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int v23; // esi
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // ebx
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 *v37; // rbx
  char v38; // cl
  __int64 v39; // rbx
  __int64 v40; // rbx
  __int64 *v41; // rdi
  __int64 v42; // r12
  void *v43; // rbx
  __int64 *v44; // rdi
  __int64 v45; // r12
  void *v46; // rbx
  __int64 *v47; // rdi
  __int64 v48; // rbx
  __int64 *v49; // rdi
  __int64 v50; // rbx
  __int64 v51; // rbx
  __int64 v52; // r12
  unsigned int v53; // r13d
  __int64 v54; // rdi
  __int64 v55; // r14
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // r12
  unsigned int v59; // r13d
  __int64 v60; // rdi
  unsigned int v61; // eax
  unsigned int v62; // eax

  switch ( a3 )
  {
    case 0:
      v51 = *(_QWORD *)(a2 + 88);
      v12 = (__int64 *)(v51 + 32);
      if ( *(void **)(v51 + 32) == sub_16982C0() )
        v12 = (__int64 *)(*(_QWORD *)(v51 + 40) + 8LL);
      return sub_169D890(v12) == 1.0;
    case 1:
      v40 = *(_QWORD *)(a2 + 88);
      v41 = (__int64 *)(v40 + 32);
      if ( *(void **)(v40 + 32) == sub_16982C0() )
        v41 = (__int64 *)(*(_QWORD *)(v40 + 40) + 8LL);
      return sub_169D890(v41) == 0.0;
    case 2:
      v50 = *(_QWORD *)(a2 + 88);
      v47 = (__int64 *)(v50 + 32);
      if ( *(void **)(v50 + 32) == sub_16982C0() )
        v47 = (__int64 *)(*(_QWORD *)(v50 + 40) + 8LL);
      return sub_169D8E0(v47) == 1.0;
    case 3:
      v48 = *(_QWORD *)(a2 + 88);
      v49 = (__int64 *)(v48 + 32);
      if ( *(void **)(v48 + 32) == sub_16982C0() )
        v49 = (__int64 *)(*(_QWORD *)(v48 + 40) + 8LL);
      return sub_169D8E0(v49) == 0.0;
    case 4:
    case 7:
      v3 = 1;
      return sub_21C5290(a1, a2, v3);
    case 5:
    case 8:
      v3 = 3;
      return sub_21C5290(a1, a2, v3);
    case 6:
    case 9:
      v3 = 0;
      return sub_21C5290(a1, a2, v3);
    case 10:
      v5 = *(_QWORD *)(a2 + 88);
      v6 = *(_DWORD *)(v5 + 32);
      v7 = *(__int64 **)(v5 + 24);
      if ( v6 <= 0x40 )
      {
        v8 = 64 - v6;
        result = 0;
        v9 = (__int64)((_QWORD)v7 << v8) >> v8;
        if ( v9 < 0 )
          return result;
        return v9 <= 30;
      }
      v58 = v5 + 24;
      v59 = v6 + 1;
      v60 = v5 + 24;
      v55 = v7[(v6 - 1) >> 6] & (1LL << ((unsigned __int8)v6 - 1));
      if ( v55 )
      {
        if ( v59 - (unsigned int)sub_16A5810(v60) > 0x40 || *v7 < 0 )
          return 0;
        v61 = v59 - sub_16A5810(v58);
        goto LABEL_68;
      }
      if ( v59 - (unsigned int)sub_16A57B0(v60) > 0x40 )
        return v55 != 0;
      result = 0;
      if ( *v7 >= 0 )
      {
        v61 = v59 - sub_16A57B0(v58);
LABEL_68:
        if ( v61 <= 0x40 )
        {
          v9 = *v7;
          return v9 <= 30;
        }
        return v55 != 0;
      }
      return result;
    case 11:
      v35 = *(_QWORD *)(a2 + 88);
      v36 = *(_DWORD *)(v35 + 32);
      v37 = *(__int64 **)(v35 + 24);
      if ( v36 <= 0x40 )
      {
        v38 = 64 - v36;
        result = 0;
        v39 = (__int64)((_QWORD)v37 << v38) >> v38;
        if ( v39 < 0 )
          return result;
        return v39 <= 14;
      }
      v52 = v35 + 24;
      v53 = v36 + 1;
      v54 = v35 + 24;
      v55 = v37[(v36 - 1) >> 6] & (1LL << ((unsigned __int8)v36 - 1));
      if ( v55 )
      {
        if ( v53 - (unsigned int)sub_16A5810(v54) > 0x40 || *v37 < 0 )
          return 0;
        v62 = v53 - sub_16A5810(v52);
        goto LABEL_82;
      }
      if ( v53 - (unsigned int)sub_16A57B0(v54) > 0x40 )
        return v55 != 0;
      result = 0;
      if ( *v37 >= 0 )
      {
        v62 = v53 - sub_16A57B0(v52);
LABEL_82:
        if ( v62 <= 0x40 )
        {
          v39 = *v37;
          return v39 <= 14;
        }
        return v55 != 0;
      }
      return result;
    case 12:
      v29 = *(_QWORD *)(a2 + 88);
      v30 = *(_DWORD *)(v29 + 32);
      v31 = *(_QWORD *)(v29 + 24);
      v32 = 1LL << ((unsigned __int8)v30 - 1);
      if ( v30 > 0x40 )
      {
        v57 = v29 + 24;
        if ( (*(_QWORD *)(v31 + 8LL * ((v30 - 1) >> 6)) & v32) != 0 )
          v20 = sub_16A5810(v57);
        else
          v20 = sub_16A57B0(v57);
      }
      else if ( (v31 & v32) != 0 )
      {
        v20 = 64;
        v33 = ~(v31 << (64 - (unsigned __int8)v30));
        if ( v33 )
        {
          _BitScanReverse64(&v34, v33);
          v20 = v34 ^ 0x3F;
        }
      }
      else
      {
        v20 = *(_DWORD *)(v29 + 32);
        if ( v31 )
        {
          _BitScanReverse64(&v31, v31);
          v20 = v30 - 64 + (v31 ^ 0x3F);
        }
      }
      v18 = v30 + 1;
      return v18 - v20 <= 0x20;
    case 13:
      v22 = *(_QWORD *)(a2 + 88);
      v23 = *(_DWORD *)(v22 + 32);
      v24 = *(_QWORD *)(v22 + 24);
      v25 = 1LL << ((unsigned __int8)v23 - 1);
      if ( v23 > 0x40 )
      {
        v56 = v22 + 24;
        if ( (*(_QWORD *)(v24 + 8LL * ((v23 - 1) >> 6)) & v25) != 0 )
          v26 = sub_16A5810(v56);
        else
          v26 = sub_16A57B0(v56);
        return v23 + 1 - v26 <= 0x10;
      }
      if ( (v25 & v24) != 0 )
      {
        v26 = 64;
        v27 = ~(v24 << (64 - (unsigned __int8)v23));
        if ( v27 )
        {
          _BitScanReverse64(&v28, v27);
          v26 = v28 ^ 0x3F;
        }
        return v23 + 1 - v26 <= 0x10;
      }
      result = 1;
      if ( v24 )
      {
        _BitScanReverse64(&v24, v24);
        return 65 - ((unsigned int)v24 ^ 0x3F) <= 0x10;
      }
      return result;
    case 14:
      v17 = *(_QWORD *)(a2 + 88);
      v18 = *(_DWORD *)(v17 + 32);
      if ( v18 > 0x40 )
      {
        v20 = sub_16A57B0(v17 + 24);
      }
      else
      {
        v19 = *(_QWORD *)(v17 + 24);
        v20 = *(_DWORD *)(v17 + 32);
        if ( v19 )
        {
          _BitScanReverse64(&v21, v19);
          v20 = v18 - 64 + (v21 ^ 0x3F);
        }
      }
      return v18 - v20 <= 0x20;
    case 15:
      v13 = *(_QWORD *)(a2 + 88);
      v14 = *(_DWORD *)(v13 + 32);
      if ( v14 > 0x40 )
        return v14 - (unsigned int)sub_16A57B0(v13 + 24) <= 0x10;
      v15 = *(_QWORD *)(v13 + 24);
      result = 1;
      if ( v15 )
      {
        _BitScanReverse64(&v16, v15);
        return 64 - ((unsigned int)v16 ^ 0x3F) <= 0x10;
      }
      return result;
    case 16:
      return *(_BYTE *)(a2 + 88) == 5;
    case 17:
      return *(_BYTE *)(a2 + 88) == 6;
    case 18:
      v10 = *(_QWORD *)(a2 + 88);
      v11 = sub_1698270();
      result = 0;
      if ( v11 == *(void **)(v10 + 32) )
      {
        v12 = (__int64 *)(v10 + 32);
        if ( v11 == sub_16982C0() )
          v12 = (__int64 *)(*(_QWORD *)(v10 + 40) + 8LL);
        return sub_169D890(v12) == 1.0;
      }
      return result;
    case 19:
      v45 = *(_QWORD *)(a2 + 88);
      v46 = sub_1698280();
      result = 0;
      if ( v46 == *(void **)(v45 + 32) )
      {
        v47 = (__int64 *)(v45 + 32);
        if ( v46 == sub_16982C0() )
          v47 = (__int64 *)(*(_QWORD *)(v45 + 40) + 8LL);
        return sub_169D8E0(v47) == 1.0;
      }
      return result;
    case 20:
      v42 = *(_QWORD *)(a2 + 88);
      v43 = sub_1698280();
      result = 0;
      if ( v43 == *(void **)(v42 + 32) )
      {
        v44 = (__int64 *)(v42 + 32);
        if ( v43 == sub_16982C0() )
          v44 = (__int64 *)(*(_QWORD *)(v42 + 40) + 8LL);
        return sub_169D8E0(v44) == -1.0;
      }
      return result;
  }
}
