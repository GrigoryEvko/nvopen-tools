// Function: sub_3022840
// Address: 0x3022840
//
__int64 __fastcall sub_3022840(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // r8
  __int64 v9; // r14
  unsigned int v10; // r13d
  unsigned int v11; // edi
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v16; // r15
  __int64 *v17; // r12
  __int64 v18; // rdi
  int v19; // edx
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r10
  __int64 v23; // rsi
  unsigned int v24; // r9d
  __int64 *v25; // rax
  __int64 v26; // r11
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // rax
  unsigned __int8 v32; // dl
  _BYTE *v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // edx
  int v37; // eax
  int v38; // eax
  int v39; // r10d
  int v40; // r9d
  int v41; // eax
  int v42; // r10d
  __int64 v43; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_52:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208AC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_52;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208AC);
  v7 = *(_DWORD *)(v6 + 224);
  v8 = *(_QWORD *)(v6 + 208);
  v9 = v6;
  if ( !v7 )
    return 0;
  v10 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v11 = (v7 - 1) & v10;
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v41 = 1;
    while ( v13 != -4096 )
    {
      v42 = v41 + 1;
      v11 = (v7 - 1) & (v41 + v11);
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_7;
      v41 = v42;
    }
    return 0;
  }
LABEL_7:
  v14 = v12[1];
  if ( !v14 )
    return 0;
  if ( a2 != **(_QWORD **)(v14 + 32) )
    return 0;
  v16 = *(__int64 **)(a2 + 64);
  v17 = &v16[*(unsigned int *)(a2 + 72)];
  if ( v17 == v16 )
    return 0;
  while ( 1 )
  {
    v18 = *v16;
    if ( !v7 )
      goto LABEL_32;
    v19 = v7 - 1;
    v20 = v19 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v21 = (__int64 *)(v8 + 16LL * v20);
    v22 = *v21;
    if ( v18 == *v21 )
    {
LABEL_16:
      v23 = v21[1];
    }
    else
    {
      v38 = 1;
      while ( v22 != -4096 )
      {
        v40 = v38 + 1;
        v20 = v19 & (v38 + v20);
        v21 = (__int64 *)(v8 + 16LL * v20);
        v22 = *v21;
        if ( v18 == *v21 )
          goto LABEL_16;
        v38 = v40;
      }
      v23 = 0;
    }
    v24 = v19 & v10;
    v25 = (__int64 *)(v8 + 16LL * (v19 & v10));
    v26 = *v25;
    if ( a2 == *v25 )
    {
LABEL_18:
      v27 = v25[1];
    }
    else
    {
      v37 = 1;
      while ( v26 != -4096 )
      {
        v39 = v37 + 1;
        v24 = v19 & (v37 + v24);
        v25 = (__int64 *)(v8 + 16LL * v24);
        v26 = *v25;
        if ( a2 == *v25 )
          goto LABEL_18;
        v37 = v39;
      }
      v27 = 0;
    }
    if ( v27 == v23 )
    {
LABEL_32:
      v28 = *(_QWORD *)(v18 + 16);
      if ( v28 )
      {
        v29 = *(_QWORD *)(v28 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v29 == v28 + 48 )
          goto LABEL_51;
        if ( !v29 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
LABEL_51:
          BUG();
        if ( (*(_BYTE *)(v29 - 17) & 0x20) != 0 )
        {
          v30 = sub_B91C10(v29 - 24, 18);
          if ( v30 )
          {
            v43 = v30;
            if ( sub_2A11940(v30, "llvm.loop.unroll.disable", 0x18u) )
              return 1;
            v31 = sub_2A11940(v43, "llvm.loop.unroll.count", 0x16u);
            if ( v31 )
            {
              v32 = *(v31 - 16);
              v33 = (v32 & 2) != 0 ? (_BYTE *)*((_QWORD *)v31 - 4) : &v31[-16 - 8LL * ((v32 >> 2) & 0xF)];
              v34 = *(_QWORD *)(*((_QWORD *)v33 + 1) + 136LL);
              v35 = *(_DWORD *)(v34 + 32);
              if ( v35 <= 0x40 ? *(_QWORD *)(v34 + 24) == 1 : v35 - 1 == (unsigned int)sub_C444A0(v34 + 24) )
                return 1;
            }
          }
        }
      }
    }
    if ( v17 == ++v16 )
      return 0;
    v8 = *(_QWORD *)(v9 + 208);
    v7 = *(_DWORD *)(v9 + 224);
  }
}
