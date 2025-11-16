// Function: sub_F95210
// Address: 0xf95210
//
__int64 __fastcall sub_F95210(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // rcx
  int v12; // edx
  int v13; // r13d
  __int64 v14; // rsi
  __int64 v15; // rax
  char *v16; // r11
  __int64 v17; // r9
  size_t v18; // rdx
  int v19; // eax
  char *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // r13
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  int v30; // eax
  int v31; // ecx
  __int64 result; // rax
  unsigned __int64 v33; // rcx
  char *v34; // r11
  __int64 v35; // rbx
  char *v36; // r14
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v43; // [rsp+0h] [rbp-150h]
  __int64 v44; // [rsp+0h] [rbp-150h]
  char *v45; // [rsp+8h] [rbp-148h]
  __int64 v46; // [rsp+8h] [rbp-148h]
  __int64 v47; // [rsp+8h] [rbp-148h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  char *v49; // [rsp+18h] [rbp-138h]
  __int64 v50; // [rsp+18h] [rbp-138h]
  char *v51; // [rsp+18h] [rbp-138h]
  char *v52; // [rsp+18h] [rbp-138h]
  __int64 *v54; // [rsp+28h] [rbp-128h]
  unsigned __int8 v55; // [rsp+28h] [rbp-128h]
  __int64 v56; // [rsp+30h] [rbp-120h] BYREF
  unsigned int v57; // [rsp+38h] [rbp-118h]
  int v58; // [rsp+3Ch] [rbp-114h]
  _QWORD v59[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v60[4]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v61; // [rsp+70h] [rbp-E0h]
  void *base; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v63; // [rsp+88h] [rbp-C8h]
  char v64; // [rsp+90h] [rbp-C0h] BYREF
  char *v65; // [rsp+98h] [rbp-B8h]
  char v66; // [rsp+A8h] [rbp-A8h] BYREF
  char *v67; // [rsp+C8h] [rbp-88h]
  char v68; // [rsp+D8h] [rbp-78h] BYREF

  v7 = **(_QWORD **)(a1 - 8);
  v8 = sub_BD5C60(a1);
  v9 = *(_QWORD *)(v7 + 8);
  v54 = (__int64 *)v8;
  if ( *(_DWORD *)(v9 + 8) > 0x40FFu )
    return 0;
  v10 = *(unsigned __int8 **)(a3 + 32);
  v11 = &v10[*(_QWORD *)(a3 + 40)];
  if ( v11 == v10 )
    return 0;
  while ( (unsigned int)*v10 < *(_DWORD *)(v9 + 8) >> 8 )
  {
    if ( v11 == ++v10 )
      return 0;
  }
  v60[0] = v7;
  v60[1] = sub_ACD6D0(v54);
  sub_DF8D10((__int64)&base, 67, v9, (char *)v60, 2);
  v48 = sub_DFD690(a4, (__int64)&base);
  v13 = v12;
  if ( v67 != &v68 )
    _libc_free(v67, &base);
  if ( v65 != &v66 )
    _libc_free(v65, &base);
  if ( v13 )
  {
    if ( v13 <= 0 )
      goto LABEL_12;
    return 0;
  }
  if ( v48 > 1 )
    return 0;
LABEL_12:
  if ( ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1 <= 3 )
    return 0;
  v14 = 1;
  v15 = sub_AA5030(*(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL), 1);
  if ( !v15 )
    BUG();
  if ( *(_BYTE *)(v15 - 24) != 36 )
    return 0;
  v16 = &v64;
  v17 = 0;
  v18 = 0;
  v63 = 0x400000000LL;
  v19 = *(_DWORD *)(a1 + 4);
  v20 = &v64;
  base = &v64;
  v21 = ((v19 & 0x7FFFFFFu) >> 1) - 1;
  if ( v21 )
  {
    do
    {
      v22 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(2 * ++v17));
      v23 = *(_QWORD *)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
        v23 = *(_QWORD *)v23;
      if ( !v23 || (v23 & (v23 - 1)) != 0 )
      {
        v20 = (char *)base;
        result = 0;
        goto LABEL_44;
      }
      v24 = (unsigned int)v63;
      v25 = (unsigned int)v63 + 1LL;
      if ( v25 > HIDWORD(v63) )
      {
        v14 = (__int64)v16;
        v44 = v21;
        v47 = v17;
        v51 = v16;
        sub_C8D5F0((__int64)&base, v16, (unsigned int)v63 + 1LL, 8u, v25, v17);
        v24 = (unsigned int)v63;
        v21 = v44;
        v17 = v47;
        v16 = v51;
      }
      *((_QWORD *)base + v24) = v23;
      v18 = (unsigned int)(v63 + 1);
      LODWORD(v63) = v63 + 1;
    }
    while ( v21 != v17 );
    v20 = (char *)base;
    v21 = 8 * v18;
    if ( v18 > 1 )
    {
      v49 = v16;
      qsort(base, v18, 8u, (__compar_fn_t)sub_A15280);
      v18 = (unsigned int)v63;
      v20 = (char *)base;
      v16 = v49;
      v21 = 8LL * (unsigned int)v63;
    }
  }
  _RAX = *(_QWORD *)&v20[v21 - 8];
  _RCX = *(_QWORD *)v20;
  if ( _RAX )
  {
    __asm { tzcnt   rax, rax }
    LODWORD(_RSI) = 64;
    if ( !_RCX )
      goto LABEL_28;
  }
  else
  {
    if ( !_RCX )
    {
      v33 = 1;
LABEL_29:
      result = 0;
      v14 = 100 * v18;
      if ( 100 * v18 >= 40 * v33 )
      {
        v45 = v16;
        sub_D5F1F0(a2, a1);
        v34 = v45;
        v50 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) - 1;
        if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1 != 1 )
        {
          v46 = v7;
          v35 = 0;
          v36 = v34;
          do
          {
            v40 = 32LL * (unsigned int)(2 * ++v35);
            v38 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + v40);
            v37 = *(_DWORD *)(v38 + 32);
            if ( v37 > 0x40 )
            {
              v43 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(2 * v35));
              v37 = sub_C44590(v38 + 24);
              v38 = v43;
            }
            else
            {
              _RDI = *(_QWORD *)(v38 + 24);
              __asm { tzcnt   rsi, rdi }
              if ( !_RDI )
                LODWORD(_RSI) = 64;
              if ( v37 > (unsigned int)_RSI )
                v37 = _RSI;
            }
            v39 = sub_ACD640(*(_QWORD *)(v38 + 8), v37, 0);
            sub_AC2B30(v40 + *(_QWORD *)(a1 - 8), v39);
          }
          while ( v50 != v35 );
          v7 = v46;
          v34 = v36;
        }
        v52 = v34;
        v61 = 257;
        v58 = 0;
        v59[0] = v7;
        v59[1] = sub_ACD6D0(v54);
        v56 = v9;
        v14 = sub_B33D10(a2, 0x43u, (__int64)&v56, 1, (int)v59, 2, v57, (__int64)v60);
        sub_AC2B30(*(_QWORD *)(a1 - 8), v14);
        result = 1;
        v20 = (char *)base;
        v16 = v52;
      }
      goto LABEL_44;
    }
    LODWORD(_RAX) = 64;
  }
  __asm { tzcnt   rsi, rcx }
LABEL_28:
  v30 = _RAX - _RSI;
  v14 = 0x28F5C28F5C28F5BLL;
  v31 = v30;
  result = 0;
  v33 = v31 + 1;
  if ( v33 <= 0x28F5C28F5C28F5BLL )
    goto LABEL_29;
LABEL_44:
  if ( v20 != v16 )
  {
    v55 = result;
    _libc_free(v20, v14);
    return v55;
  }
  return result;
}
