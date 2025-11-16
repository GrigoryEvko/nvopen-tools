// Function: sub_19D3C80
// Address: 0x19d3c80
//
__int64 __fastcall sub_19D3C80(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned int v8; // r15d
  unsigned int v9; // r14d
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // edx
  unsigned __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  _QWORD *v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  int v27; // r14d
  unsigned int v28; // eax
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // r8
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // [rsp+2B0h] [rbp-A0h]
  unsigned __int64 v36; // [rsp+2B8h] [rbp-98h]
  __m128i v37; // [rsp+2D0h] [rbp-80h] BYREF

  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 + 24 * (3 - v6));
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
  {
    if ( *(_QWORD *)(v7 + 24) )
      return 0;
LABEL_5:
    v11 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1 - v6)));
    if ( v11 == sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
    {
      sub_14191F0(*a1, a2);
      sub_15F20C0((_QWORD *)a2);
      return 0;
    }
    v12 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
    v13 = v12;
    if ( *(_BYTE *)(v12 + 16) == 3 && (*(_BYTE *)(v12 + 80) & 1) != 0 && !sub_15E4F60(v12) )
      __asm { jmp     rax }
    v14 = sub_141C430(*a1, a2, 0);
    if ( (v14 & 7) != 1 || (v24 = v14 & 0xFFFFFFFFFFFFFFF8LL, v25 = v24, *(_BYTE *)(v24 + 16) != 78) )
    {
      v15 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v15 + 16) != 13 )
        return 0;
      goto LABEL_17;
    }
    v26 = *(_QWORD *)(v24 - 24);
    if ( !*(_BYTE *)(v26 + 16) && (*(_BYTE *)(v26 + 33) & 0x20) != 0 && *(_DWORD *)(v26 + 36) == 137 )
    {
      v9 = sub_19D3C00(a1, a2, v25, a3, a4, a5);
      if ( (_BYTE)v9 )
        return v9;
      v15 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v15 + 16) != 13 )
        return 0;
      if ( *(_BYTE *)(v25 + 16) != 78 )
        goto LABEL_17;
    }
    else
    {
      v15 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v15 + 16) != 13 )
        return 0;
    }
    v27 = sub_15603A0((_QWORD *)(a2 + 56), 1);
    v28 = sub_15603A0((_QWORD *)(a2 + 56), 0);
    v29 = (v27 | v28) & (unsigned __int64)-(__int64)(v27 | v28);
    if ( *(_DWORD *)(v15 + 32) <= 0x40u )
      v30 = *(_QWORD *)(v15 + 24);
    else
      v30 = **(_QWORD **)(v15 + 24);
    v35 = v30;
    v36 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
    v31 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v9 = sub_19D2360((__int64)a1, a2, v31, v36, v35, v29, v25);
    if ( (_BYTE)v9 )
      goto LABEL_38;
LABEL_17:
    sub_141F730(&v37, a2);
    v16 = sub_141C340(*a1, &v37, 1u, (_QWORD *)(a2 + 24), *(_QWORD *)(a2 + 40), 0, 0, 0);
    v17 = v16 & 7;
    if ( v17 != 1 )
    {
      if ( v17 == 2 )
      {
        v18 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        v19 = *(_BYTE *)(v18 + 16);
        if ( v19 == 53 )
          goto LABEL_30;
        if ( v19 == 78 )
        {
          v20 = *(_QWORD *)(v18 - 24);
          if ( !*(_BYTE *)(v20 + 16) && (*(_BYTE *)(v20 + 33) & 0x20) != 0 && *(_DWORD *)(v20 + 36) == 117 )
          {
            v21 = *(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v21 + 16) == 13 )
            {
              v22 = *(_DWORD *)(v21 + 32) <= 0x40u ? *(_QWORD *)(v21 + 24) : **(_QWORD **)(v21 + 24);
              v23 = *(_QWORD **)(v15 + 24);
              if ( *(_DWORD *)(v15 + 32) > 0x40u )
                v23 = (_QWORD *)*v23;
              if ( (unsigned __int64)v23 <= v22 )
              {
LABEL_30:
                v9 = 1;
                sub_14191F0(*a1, a2);
                sub_15F20C0((_QWORD *)a2);
                return v9;
              }
            }
          }
        }
      }
      return 0;
    }
    v32 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v33 = v32;
    if ( *(_BYTE *)(v32 + 16) == 78 )
    {
      v34 = *(_QWORD *)(v32 - 24);
      if ( !*(_BYTE *)(v34 + 16) && (*(_BYTE *)(v34 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v34 + 36) == 133 )
          return (unsigned int)sub_19D1320((__int64)a1, a2, v33);
        if ( (*(_BYTE *)(v34 + 33) & 0x20) != 0 && *(_DWORD *)(v34 + 36) == 137 )
        {
          v9 = sub_19D1880((__int64)a1, a2, v33);
          if ( (_BYTE)v9 )
          {
LABEL_38:
            sub_14191F0(*a1, a2);
            sub_15F20C0((_QWORD *)a2);
            return v9;
          }
        }
      }
    }
    return 0;
  }
  v9 = 0;
  if ( v8 == (unsigned int)sub_16A57B0(v7 + 24) )
    goto LABEL_5;
  return v9;
}
