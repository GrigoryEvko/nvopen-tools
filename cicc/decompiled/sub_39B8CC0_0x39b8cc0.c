// Function: sub_39B8CC0
// Address: 0x39b8cc0
//
__int64 __fastcall sub_39B8CC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 **v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rcx
  int v11; // esi
  int v12; // esi
  __int64 v13; // rdi
  __int64 (*v14)(); // r9
  __int64 result; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 **v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  _BYTE *v23; // r15
  unsigned __int64 v24; // rax
  int v25; // r9d
  _QWORD *v26; // r8
  _QWORD *v27; // rbx
  __int64 v28; // rcx
  _QWORD *v29; // rax
  unsigned __int64 v30; // r12
  int v31; // edx
  __int64 v32; // rbx
  unsigned __int64 v33; // rax
  int v34; // eax
  __int64 v35; // [rsp+0h] [rbp-A0h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  unsigned int v37; // [rsp+8h] [rbp-98h]
  _QWORD *v38; // [rsp+8h] [rbp-98h]
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
      v35 = a4;
      v36 = a3;
      if ( (unsigned __int8)sub_15F8F00(a2) )
        return 0;
      v6 = *(_BYTE *)(a2 + 16);
      a3 = v36;
      a4 = v35;
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
            return sub_39B3AF0(a1, (_QWORD *)a2, *(_QWORD **)(a3 + 8 * a4 - 8));
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
    v22 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_21;
    if ( (v16 & 4) != 0 )
    {
      v23 = *(_BYTE **)(v22 - 24);
      if ( !v23[16] )
        goto LABEL_39;
    }
    else
    {
      v23 = *(_BYTE **)(v22 - 72);
      if ( !v23[16] )
      {
LABEL_39:
        v24 = sub_134EF80(&v39);
        v40 = v42;
        v26 = (_QWORD *)v24;
        v27 = (_QWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL)
                       - 24LL * (*(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
        v28 = v24 - (_QWORD)v27;
        v41 = 0x800000000LL;
        v29 = v42;
        v30 = 0xAAAAAAAAAAAAAAABLL * (v28 >> 3);
        v31 = 0;
        if ( (unsigned __int64)v28 > 0xC0 )
        {
          v38 = v26;
          sub_16CD150((__int64)&v40, v42, 0xAAAAAAAAAAAAAAABLL * (v28 >> 3), 8, (int)v26, v25);
          v31 = v41;
          v26 = v38;
          v29 = &v40[8 * (unsigned int)v41];
        }
        if ( v26 != v27 )
        {
          do
          {
            if ( v29 )
              *v29 = *v27;
            v27 += 3;
            ++v29;
          }
          while ( v26 != v27 );
          v31 = v41;
        }
        LODWORD(v41) = v30 + v31;
        result = sub_39B3D80((__int64)a1, (__int64)v23, (int)v30 + v31);
        if ( v40 != v42 )
        {
          v37 = result;
          _libc_free((unsigned __int64)v40);
          return v37;
        }
        return result;
      }
    }
    v32 = **(_QWORD **)(*(_QWORD *)v23 + 16LL);
    v33 = sub_134EF80(&v39);
    v34 = -1431655765
        * ((__int64)(v33
                   - ((v39 & 0xFFFFFFFFFFFFFFF8LL)
                    - 24LL * (*(_DWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
    if ( v34 < 0 )
      v34 = *(_DWORD *)(v32 + 12) - 1;
    return (unsigned int)(v34 + 1);
  }
LABEL_2:
  if ( v6 == 5 && *(_WORD *)(a2 + 18) == 32 )
  {
LABEL_26:
    v17 = a4 - 1;
    v18 = a3 + 8;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v19 = *(__int64 ***)(a2 - 8);
    else
      v19 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v20 = *v19;
    v21 = sub_16348C0(a2);
    return sub_39B8230(a1, v21, v20, v18, v17);
  }
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
    goto LABEL_61;
  }
  v12 = *(unsigned __int16 *)(a2 + 18);
LABEL_11:
  v13 = a1[2];
  if ( v12 == 36 )
  {
    v14 = *(__int64 (**)())(*(_QWORD *)v13 + 784LL);
    result = 1;
    if ( v14 != sub_1D5A3F0 )
      return ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v14)(v13, v10, v9) ^ 1u;
    return result;
  }
  if ( v12 != 37 )
  {
LABEL_61:
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
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 816LL);
  result = 1;
  if ( v14 != sub_1D5A400 )
    return ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v14)(v13, v10, v9) ^ 1u;
  return result;
}
