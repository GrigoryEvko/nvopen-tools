// Function: sub_170BD90
// Address: 0x170bd90
//
__int64 __fastcall sub_170BD90(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r15
  char v5; // al
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v12; // r14
  _QWORD *v13; // rax
  __int64 **v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r15
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // rbx
  __int64 v19; // rdi
  unsigned __int64 *v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // rbx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r13
  __int64 v32; // r14
  __int64 v33; // rax
  _BYTE *v34; // r15
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rcx
  int v38; // eax
  __int64 v39; // rax
  unsigned int v40; // r15d
  bool v41; // al
  unsigned int v42; // edx
  __int64 v43; // rax
  char v44; // si
  unsigned int v45; // edx
  int v46; // eax
  int v47; // [rsp+0h] [rbp-70h]
  unsigned int v48; // [rsp+4h] [rbp-6Ch]
  __int64 v49; // [rsp+8h] [rbp-68h]
  int v50; // [rsp+8h] [rbp-68h]
  int v51; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v52; // [rsp+18h] [rbp-58h] BYREF
  __int64 v53[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v54; // [rsp+30h] [rbp-40h]

  v3 = (_QWORD *)a2;
  v4 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 == 9 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v13 = (_QWORD *)sub_16498A0(a2);
    v14 = (__int64 **)sub_16471A0(v13, 0);
    v49 = sub_1599EF0(v14);
    v15 = (__int64 *)sub_16498A0(a2);
    v16 = sub_159C4F0(v15);
    v54 = 257;
    v17 = (unsigned __int8 *)sub_1648A60(64, 2u);
    v18 = v17;
    if ( v17 )
      sub_15F9650((__int64)v17, v16, v49, 0, 0);
    v19 = *(_QWORD *)(v12 + 8);
    if ( v19 )
    {
      v20 = *(unsigned __int64 **)(v12 + 16);
      sub_157E9D0(v19 + 40, (__int64)v18);
      v21 = *((_QWORD *)v18 + 3);
      v22 = *v20;
      *((_QWORD *)v18 + 4) = v20;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v18 + 3) = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v18 + 24;
      *v20 = *v20 & 7 | (unsigned __int64)(v18 + 24);
    }
    sub_164B780((__int64)v18, v53);
    v52 = v18;
    if ( !*(_QWORD *)(v12 + 80) )
      sub_4263D6(v18, v53, v23);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(v12 + 88))(v12 + 64, &v52);
    v24 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 )
    {
      v52 = *(unsigned __int8 **)v12;
      sub_1623A60((__int64)&v52, v24, 2);
      v25 = *((_QWORD *)v18 + 6);
      if ( v25 )
        sub_161E7C0((__int64)(v18 + 48), v25);
      v26 = v52;
      *((_QWORD *)v18 + 6) = v52;
      if ( v26 )
        sub_1623210((__int64)&v52, v26, (__int64)(v18 + 48));
    }
  }
  else if ( v5 != 15 )
  {
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
    v6 = *(_QWORD *)(a2 + 40);
    v7 = sub_157F0B0(v6);
    if ( !v7 )
      return 0;
    v8 = *(_QWORD *)(v6 + 48);
    if ( v6 + 40 == v8 )
      return 0;
    v9 = 0;
    do
    {
      v8 = *(_QWORD *)(v8 + 8);
      ++v9;
    }
    while ( v6 + 40 != v8 );
    if ( v9 != 2 )
      return 0;
    v10 = sub_157EBA0(v6);
    if ( *(_BYTE *)(v10 + 16) != 26 )
      return 0;
    if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 1 )
      return 0;
    v27 = *(_QWORD *)(v10 - 24);
    v28 = sub_157EBA0(v7);
    v31 = v28;
    if ( *(_BYTE *)(v28 + 16) != 26 )
      return 0;
    if ( (*(_DWORD *)(v28 + 20) & 0xFFFFFFF) != 3 )
      return 0;
    v32 = *(_QWORD *)(v28 - 72);
    if ( *(_BYTE *)(v32 + 16) != 75 )
      return 0;
    v33 = *(_QWORD *)(v32 - 48);
    if ( v4 != v33 )
      return 0;
    if ( !v33 )
      return 0;
    v34 = *(_BYTE **)(v32 - 24);
    if ( v34[16] > 0x10u )
      return 0;
    if ( !sub_1593BB0(*(_QWORD *)(v32 - 24), a2, v29, v30) )
    {
      if ( v34[16] == 13 )
      {
        if ( *((_DWORD *)v34 + 8) <= 0x40u )
        {
          if ( *((_QWORD *)v34 + 3) )
            return 0;
        }
        else
        {
          v50 = *((_DWORD *)v34 + 8);
          if ( v50 != (unsigned int)sub_16A57B0((__int64)(v34 + 24)) )
            return 0;
        }
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) != 16 )
          return 0;
        v39 = sub_15A1020(v34, a2, v35, v36);
        if ( v39 && *(_BYTE *)(v39 + 16) == 13 )
        {
          v40 = *(_DWORD *)(v39 + 32);
          if ( v40 <= 0x40 )
            v41 = *(_QWORD *)(v39 + 24) == 0;
          else
            v41 = v40 == (unsigned int)sub_16A57B0(v39 + 24);
          if ( !v41 )
            return 0;
        }
        else
        {
          v42 = 0;
          v51 = *(_DWORD *)(*(_QWORD *)v34 + 32LL);
          while ( v51 != v42 )
          {
            v48 = v42;
            v43 = sub_15A0A60((__int64)v34, v42);
            if ( !v43 )
              return 0;
            v44 = *(_BYTE *)(v43 + 16);
            v45 = v48;
            if ( v44 != 9 )
            {
              if ( v44 != 13 )
                return 0;
              if ( *(_DWORD *)(v43 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v43 + 24) )
                  return 0;
              }
              else
              {
                v47 = *(_DWORD *)(v43 + 32);
                v46 = sub_16A57B0(v43 + 24);
                v45 = v48;
                if ( v47 != v46 )
                  return 0;
              }
            }
            v42 = v45 + 1;
          }
        }
      }
    }
    v38 = *(unsigned __int16 *)(v32 + 18);
    v37 = *(_QWORD *)(v31 - 24);
    BYTE1(v38) &= ~0x80u;
    if ( (unsigned int)(v38 - 32) <= 1 )
    {
      if ( v38 != 32 )
        v37 = *(_QWORD *)(v31 - 48);
      if ( v37 == v27 )
      {
        sub_15F22F0(v3, v31);
        return (__int64)v3;
      }
    }
    return 0;
  }
  return sub_170BC50(a1, (__int64)v3);
}
