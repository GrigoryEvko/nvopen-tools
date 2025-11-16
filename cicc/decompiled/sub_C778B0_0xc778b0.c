// Function: sub_C778B0
// Address: 0xc778b0
//
__int64 __fastcall sub_C778B0(__int64 a1, _QWORD *a2, char a3)
{
  unsigned int v5; // r13d
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v10; // edx
  const void **v12; // r8
  __int64 v13; // rax
  unsigned int v14; // r13d
  unsigned int v15; // eax
  unsigned int v16; // r15d
  int v19; // r13d
  unsigned __int64 v20; // rax
  unsigned int v21; // r13d
  unsigned int v22; // r14d
  unsigned int v23; // r14d
  __int64 v24; // rax
  unsigned int v25; // edx
  __int64 v26; // rsi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned int v30; // eax
  unsigned int v33; // eax
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned int v37; // eax
  unsigned int v38; // r13d
  unsigned int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned int v42; // r14d
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // r13d
  unsigned int v47; // ebx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // [rsp+8h] [rbp-D8h]
  __int64 v51; // [rsp+8h] [rbp-D8h]
  const void **v52; // [rsp+8h] [rbp-D8h]
  _QWORD *v53; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v54; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v55; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v56; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v57; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v59; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-98h]
  unsigned __int64 v61; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-88h]
  __int64 v63; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-78h]
  unsigned __int64 v65; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-68h]
  _QWORD *v67; // [rsp+80h] [rbp-60h]
  unsigned int v68; // [rsp+88h] [rbp-58h]
  _QWORD *v69; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v70; // [rsp+98h] [rbp-48h]
  _QWORD *v71; // [rsp+A0h] [rbp-40h]
  unsigned int v72; // [rsp+A8h] [rbp-38h]

  v5 = *((_DWORD *)a2 + 2);
  v6 = *a2;
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 > 0x40 )
  {
    if ( (*(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6)) & v7) != 0 )
    {
      *(_DWORD *)(a1 + 8) = v5;
      sub_C43780(a1, (const void **)a2);
      v30 = *((_DWORD *)a2 + 6);
      *(_DWORD *)(a1 + 24) = v30;
      if ( v30 <= 0x40 )
        goto LABEL_4;
LABEL_44:
      sub_C43780(a1 + 16, (const void **)a2 + 2);
      return a1;
    }
    v58 = *((_DWORD *)a2 + 2);
    sub_C43690((__int64)&v57, 0, 0);
    v60 = v5;
    sub_C43690((__int64)&v59, 0, 0);
    v5 = *((_DWORD *)a2 + 2);
LABEL_7:
    v10 = *((_DWORD *)a2 + 6);
    _RSI = a2[2];
    v12 = (const void **)(a2 + 2);
    v13 = 1LL << ((unsigned __int8)v10 - 1);
    if ( v10 > 0x40 )
    {
      if ( (*(_QWORD *)(_RSI + 8LL * ((v10 - 1) >> 6)) & v13) != 0 )
      {
LABEL_9:
        v62 = v5;
        if ( v5 > 0x40 )
        {
          sub_C43780((__int64)&v61, (const void **)a2);
          v10 = *((_DWORD *)a2 + 6);
          v12 = (const void **)(a2 + 2);
        }
        else
        {
          v61 = *a2;
        }
        v64 = v10;
        if ( v10 > 0x40 )
          sub_C43780((__int64)&v63, v12);
        else
          v63 = a2[2];
        v14 = *((_DWORD *)a2 + 2);
        if ( a3 )
        {
          if ( v14 > 0x40 )
          {
            if ( v14 != (unsigned int)sub_C44630((__int64)a2) + 2 )
            {
              v54 = v14;
              goto LABEL_100;
            }
            v33 = sub_C445E0((__int64)a2);
            v34 = 1LL << v33;
          }
          else
          {
            v50 = *a2;
            if ( v14 != (unsigned int)sub_39FAC40(*a2) + 2 )
            {
              v54 = v14;
LABEL_50:
              v53 = 0;
              goto LABEL_51;
            }
            _RDX = ~v50;
            if ( v50 == -1 )
            {
              v34 = 1;
              v33 = 64;
            }
            else
            {
              __asm { tzcnt   rcx, rdx }
              v33 = _RCX;
              v34 = 1LL << _RCX;
            }
          }
          if ( v64 > 0x40 )
          {
            *(_QWORD *)(v63 + 8LL * (v33 >> 6)) |= v34;
            v14 = *((_DWORD *)a2 + 2);
          }
          else
          {
            v63 |= v34;
          }
        }
        v54 = v14;
        if ( v14 <= 0x40 )
          goto LABEL_50;
LABEL_100:
        sub_C43690((__int64)&v53, 0, 0);
        v14 = v54;
LABEL_51:
        v70 = v14;
        if ( v14 > 0x40 )
        {
          sub_C43780((__int64)&v69, (const void **)&v53);
          v14 = v54;
          v56 = v54;
          if ( v54 > 0x40 )
          {
            sub_C43780((__int64)&v55, (const void **)&v53);
            v14 = v56;
            if ( v56 > 0x40 )
            {
              sub_C43D10((__int64)&v55);
              v14 = v56;
              v36 = v55;
LABEL_56:
              v65 = v36;
              v68 = v70;
              v66 = v14;
              v67 = v69;
              sub_C70430((__int64)&v69, 0, a3, 0, (__int64)&v65, (__int64)&v61);
              if ( v58 > 0x40 && v57 )
                j_j___libc_free_0_0(v57);
              v57 = (unsigned __int64)v69;
              v37 = v70;
              v70 = 0;
              v58 = v37;
              if ( v60 > 0x40 && v59 )
              {
                j_j___libc_free_0_0(v59);
                v59 = (unsigned __int64)v71;
                v60 = v72;
                if ( v70 > 0x40 && v69 )
                  j_j___libc_free_0_0(v69);
              }
              else
              {
                v59 = (unsigned __int64)v71;
                v60 = v72;
              }
              if ( v68 > 0x40 && v67 )
                j_j___libc_free_0_0(v67);
              if ( v66 > 0x40 && v65 )
                j_j___libc_free_0_0(v65);
              if ( v54 > 0x40 && v53 )
                j_j___libc_free_0_0(v53);
              v38 = v64;
              if ( !a3 )
                goto LABEL_83;
              if ( v64 > 0x40 )
              {
                if ( (unsigned int)sub_C44630((__int64)&v63) != 1 )
                  goto LABEL_84;
              }
              else if ( (unsigned int)sub_39FAC40(v63) != 1 )
              {
LABEL_76:
                if ( v62 > 0x40 && v61 )
                  j_j___libc_free_0_0(v61);
                goto LABEL_79;
              }
              v42 = v62;
              v43 = v62 > 0x40 ? sub_C44630((__int64)&v61) : sub_39FAC40(v61);
              if ( v42 - v43 == 1 )
              {
LABEL_83:
                if ( v38 <= 0x40 )
                  goto LABEL_76;
LABEL_84:
                if ( v63 )
                  j_j___libc_free_0_0(v63);
                goto LABEL_76;
              }
              v44 = ~(1LL << ((unsigned __int8)v38 - 1));
              if ( v38 > 0x40 )
              {
                *(_QWORD *)(v63 + 8LL * ((v38 - 1) >> 6)) &= v44;
                v42 = v62;
              }
              else
              {
                v63 &= v44;
              }
              v45 = 1LL << ((unsigned __int8)v42 - 1);
              if ( v42 > 0x40 )
              {
                *(_QWORD *)(v61 + 8LL * ((v42 - 1) >> 6)) |= v45;
                v46 = *((_DWORD *)a2 + 2);
                v42 = v62;
                v47 = v46 - 1;
                if ( v62 > 0x40 )
                {
                  v46 -= sub_C44500((__int64)&v61);
                  goto LABEL_114;
                }
              }
              else
              {
                v46 = *((_DWORD *)a2 + 2);
                v61 |= v45;
                v47 = v46 - 1;
              }
              if ( v42 )
              {
                if ( v61 << (64 - (unsigned __int8)v42) == -1 )
                {
                  v46 -= 64;
                }
                else
                {
                  _BitScanReverse64(&v48, ~(v61 << (64 - (unsigned __int8)v42)));
                  v46 -= v48 ^ 0x3F;
                }
              }
LABEL_114:
              if ( v46 != v47 )
              {
                if ( v46 <= 0x3F && v47 <= 0x40 )
                {
                  v49 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v46 + 64 - (unsigned __int8)v47) << v46;
                  if ( v60 > 0x40 )
                  {
                    *(_QWORD *)v59 |= v49;
                    v38 = v64;
                  }
                  else
                  {
                    v38 = v64;
                    v59 |= v49;
                  }
                  goto LABEL_83;
                }
                sub_C43C90(&v59, v46, v47);
              }
              v38 = v64;
              goto LABEL_83;
            }
            v35 = v55;
LABEL_54:
            v36 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v35;
            if ( !v14 )
              v36 = 0;
            goto LABEL_56;
          }
        }
        else
        {
          v69 = v53;
        }
        v35 = (unsigned __int64)v53;
        goto LABEL_54;
      }
      v15 = sub_C44590((__int64)(a2 + 2));
      v12 = (const void **)(a2 + 2);
      v16 = v15;
    }
    else
    {
      if ( (v13 & _RSI) != 0 )
        goto LABEL_9;
      v16 = 64;
      __asm { tzcnt   rax, rsi }
      if ( _RSI )
        v16 = _RAX;
      if ( v10 <= v16 )
        v16 = v10;
    }
    if ( v5 > 0x40 )
    {
      v52 = v12;
      v39 = sub_C445E0((__int64)a2);
      v12 = v52;
      v19 = v39;
      if ( !v39 )
        goto LABEL_27;
      if ( v39 > 0x40 )
      {
        sub_C43C90(&v57, 0, v39);
        v12 = v52;
        goto LABEL_27;
      }
      v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39);
    }
    else
    {
      _RAX = ~*a2;
      if ( *a2 == -1 )
      {
        v20 = -1;
        v19 = 64;
      }
      else
      {
        __asm { tzcnt   rax, rax }
        v19 = _RAX;
        if ( !(_DWORD)_RAX )
          goto LABEL_27;
        v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)_RAX);
      }
    }
    if ( v58 > 0x40 )
      *(_QWORD *)v57 |= v20;
    else
      v57 |= v20;
LABEL_27:
    if ( v16 == v19 && *((_DWORD *)a2 + 2) > v16 )
    {
      v21 = v60;
      v41 = 1LL << v16;
      if ( v60 <= 0x40 )
      {
        v59 |= v41;
        if ( a3 )
        {
          v24 = ~(1LL << ((unsigned __int8)v60 - 1));
          goto LABEL_33;
        }
LABEL_29:
        v22 = *((_DWORD *)a2 + 6);
        if ( v22 <= 0x40 )
        {
          v40 = a2[2];
          if ( !v40 || v40 == 1LL << ((unsigned __int8)v22 - 1) )
            goto LABEL_90;
        }
        else
        {
          v51 = (__int64)v12;
          if ( v22 == (unsigned int)sub_C444A0((__int64)v12)
            || (v23 = v22 - 1, (*(_QWORD *)(a2[2] + 8LL * (v23 >> 6)) & (1LL << v23)) != 0)
            && v23 == (unsigned int)sub_C44590(v51) )
          {
LABEL_90:
            v25 = v58;
            v27 = v57;
            goto LABEL_36;
          }
        }
LABEL_32:
        v24 = ~(1LL << ((unsigned __int8)v21 - 1));
        if ( v21 > 0x40 )
        {
          *(_QWORD *)(v59 + 8LL * ((v21 - 1) >> 6)) &= v24;
          goto LABEL_34;
        }
LABEL_33:
        v59 &= v24;
LABEL_34:
        v25 = v58;
        v26 = 1LL << ((unsigned __int8)v58 - 1);
        if ( v58 <= 0x40 )
        {
          v21 = v60;
          v27 = v26 | v57;
LABEL_36:
          *(_QWORD *)a1 = v27;
          v28 = v59;
          *(_DWORD *)(a1 + 8) = v25;
          *(_DWORD *)(a1 + 24) = v21;
          *(_QWORD *)(a1 + 16) = v28;
          return a1;
        }
        *(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6)) |= v26;
LABEL_79:
        v25 = v58;
        v27 = v57;
        v21 = v60;
        goto LABEL_36;
      }
      *(_QWORD *)(v59 + 8LL * (v16 >> 6)) |= v41;
    }
    v21 = v60;
    if ( a3 )
      goto LABEL_32;
    goto LABEL_29;
  }
  if ( (v7 & v6) == 0 )
  {
    v58 = *((_DWORD *)a2 + 2);
    v57 = 0;
    v60 = v5;
    v59 = 0;
    goto LABEL_7;
  }
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)a1 = v6;
  v8 = *((_DWORD *)a2 + 6);
  *(_DWORD *)(a1 + 24) = v8;
  if ( v8 > 0x40 )
    goto LABEL_44;
LABEL_4:
  *(_QWORD *)(a1 + 16) = a2[2];
  return a1;
}
