// Function: sub_216D6F0
// Address: 0x216d6f0
//
__int64 __fastcall sub_216D6F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 **v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rcx
  int v11; // esi
  int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // r12d
  __int64 (__fastcall *v15)(__int64, __int64, __int64); // rax
  unsigned __int64 v16; // rsi
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 **v20; // rax
  __int64 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _BYTE *v24; // r15
  unsigned __int64 v25; // rax
  int v26; // r9d
  __int64 *v27; // r8
  __int64 *v28; // rbx
  __int64 v29; // rcx
  __int64 *v30; // rax
  unsigned __int64 v31; // r12
  int v32; // edx
  __int64 v33; // rbx
  unsigned __int64 v34; // rax
  int v35; // r12d
  __int64 v36; // [rsp+0h] [rbp-A0h]
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 *v38; // [rsp+8h] [rbp-98h]
  unsigned __int64 v39; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v40; // [rsp+20h] [rbp-80h] BYREF
  __int64 v41; // [rsp+28h] [rbp-78h]
  _BYTE v42[112]; // [rsp+30h] [rbp-70h] BYREF
  __int64 savedregs; // [rsp+A0h] [rbp+0h] BYREF

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 > 0x17u )
  {
    if ( v6 == 77 || v6 == 86 )
      return 0;
    if ( v6 == 53 )
    {
      v36 = a4;
      v37 = a3;
      if ( (unsigned __int8)sub_15F8F00(a2) )
        return 0;
      v6 = *(_BYTE *)(a2 + 16);
      a3 = v37;
      a4 = v36;
      if ( v6 <= 0x17u )
        goto LABEL_2;
    }
    if ( v6 == 56 )
      goto LABEL_26;
    if ( v6 == 78 )
    {
      v16 = a2 | 4;
    }
    else
    {
      v16 = a2 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v6 != 29 )
      {
LABEL_21:
        v11 = v6;
        if ( (unsigned int)v6 - 60 > 0xC )
          goto LABEL_52;
        if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) - 75) > 1u )
        {
          if ( (unsigned __int8)(v6 - 61) <= 1u || v6 == 68 )
            return sub_2168430(a1, (_QWORD *)a2, *(_QWORD **)(a3 + 8 * a4 - 8));
LABEL_52:
          v9 = *(_QWORD *)a2;
          v7 = *(_QWORD *)a2;
          if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
          {
            v10 = 0;
            goto LABEL_10;
          }
          goto LABEL_6;
        }
        return 0;
      }
    }
    v39 = v16;
    v23 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_21;
    if ( (v16 & 4) != 0 )
    {
      v24 = *(_BYTE **)(v23 - 24);
      if ( !v24[16] )
        goto LABEL_39;
    }
    else
    {
      v24 = *(_BYTE **)(v23 - 72);
      if ( !v24[16] )
      {
LABEL_39:
        v25 = sub_134EF80(&v39);
        v40 = v42;
        v27 = (__int64 *)v25;
        v28 = (__int64 *)((v39 & 0xFFFFFFFFFFFFFFF8LL)
                        - 24LL * (*(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
        v29 = v25 - (_QWORD)v28;
        v41 = 0x800000000LL;
        v30 = (__int64 *)v42;
        v31 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 3);
        v32 = 0;
        if ( (unsigned __int64)v29 > 0xC0 )
        {
          v38 = v27;
          sub_16CD150((__int64)&v40, v42, 0xAAAAAAAAAAAAAAABLL * (v29 >> 3), 8, (int)v27, v26);
          v32 = v41;
          v27 = v38;
          v30 = (__int64 *)&v40[8 * (unsigned int)v41];
        }
        if ( v27 != v28 )
        {
          do
          {
            if ( v30 )
            {
              v29 = *v28;
              *v30 = *v28;
            }
            v28 += 3;
            ++v30;
          }
          while ( v27 != v28 );
          v32 = v41;
        }
        LODWORD(v41) = v31 + v32;
        v14 = sub_21686A0((__int64)a1, v24, (int)v31 + v32, v29);
        if ( v40 != v42 )
          _libc_free((unsigned __int64)v40);
        return v14;
      }
    }
    v33 = **(_QWORD **)(*(_QWORD *)v24 + 16LL);
    v34 = sub_134EF80(&v39);
    v35 = -1431655765
        * ((__int64)(v34
                   - ((v39 & 0xFFFFFFFFFFFFFFF8LL)
                    - 24LL * (*(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
    if ( v35 < 0 )
      v35 = *(_DWORD *)(v33 + 12) - 1;
    return (unsigned int)(v35 + 1);
  }
LABEL_2:
  if ( v6 != 5 || *(_WORD *)(a2 + 18) != 32 )
  {
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
    {
      v9 = *(_QWORD *)a2;
      v10 = 0;
      goto LABEL_55;
    }
    v7 = *(_QWORD *)a2;
LABEL_6:
    v8 = (__int64 **)(a2 - 24);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v8 = *(__int64 ***)(a2 - 8);
    v9 = v7;
    v10 = **v8;
    if ( v6 > 0x17u )
    {
      v11 = v6;
LABEL_10:
      v12 = v11 - 24;
      goto LABEL_11;
    }
LABEL_55:
    if ( v6 != 5 )
    {
      v12 = 56;
      goto LABEL_64;
    }
    v12 = *(unsigned __int16 *)(a2 + 18);
LABEL_11:
    v13 = a1[2];
    if ( v12 == 36 )
    {
      v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 784LL);
      if ( v15 == sub_2165A80 )
      {
        v14 = 1;
        if ( *(_BYTE *)(v10 + 8) == 11 && *(_BYTE *)(v9 + 8) == 11 && (unsigned int)sub_1643030(v10) == 64 )
          return (unsigned int)sub_1643030(v9) != 32;
        return v14;
      }
      return (unsigned __int8)v15(v13, v10, v9) ^ 1u;
    }
    if ( v12 == 37 )
    {
      v14 = 1;
      v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 816LL);
      if ( (char *)v15 == (char *)sub_1D5A400 )
        return v14;
      return (unsigned __int8)v15(v13, v10, v9) ^ 1u;
    }
LABEL_64:
    savedregs = (__int64)&savedregs;
    switch ( v12 )
    {
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
        JUMPOUT(0x14A1D50);
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 34:
      case 35:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
        JUMPOUT(0x14A1D38);
      case 45:
        JUMPOUT(0x14A1D90);
      case 46:
        JUMPOUT(0x14A1DE0);
      case 47:
        JUMPOUT(0x14A1D20);
      default:
        JUMPOUT(0x14A1E38);
    }
  }
LABEL_26:
  v18 = a4 - 1;
  v19 = a3 + 8;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v20 = *(__int64 ***)(a2 - 8);
  else
    v20 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v21 = *v20;
  v22 = sub_16348C0(a2);
  return sub_216CC60(a1, v22, v21, v19, v18);
}
