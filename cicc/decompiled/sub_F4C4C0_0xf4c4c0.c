// Function: sub_F4C4C0
// Address: 0xf4c4c0
//
__int64 *__fastcall sub_F4C4C0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, size_t a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 v8; // rdx
  unsigned __int8 v9; // al
  _QWORD *v10; // r12
  __int64 v11; // rcx
  _BYTE *v12; // rbx
  unsigned __int8 v13; // al
  _QWORD *v14; // rdx
  size_t v15; // rax
  _BYTE *v16; // rdi
  __int64 v17; // r8
  unsigned __int8 v18; // al
  __int64 v19; // rax
  _BYTE *v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r11
  __int64 v24; // r9
  unsigned int v25; // r8d
  _QWORD *v26; // rax
  _BYTE *v27; // rdi
  __int64 v28; // rax
  _BYTE *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rdi
  __int64 v33; // r8
  _QWORD *v34; // rdi
  size_t v35; // rdx
  size_t v36; // rdx
  _QWORD *v37; // rcx
  int v38; // eax
  int v39; // eax
  int v40; // r9d
  int v41; // r9d
  __int64 v42; // rdi
  unsigned int v43; // edx
  __int64 v44; // rsi
  int v45; // r8d
  _QWORD *v46; // r10
  int v47; // r9d
  int v48; // r9d
  __int64 v49; // rdi
  int v50; // r8d
  unsigned int v51; // edx
  __int64 v52; // rsi
  unsigned int v53; // [rsp+8h] [rbp-118h]
  __int64 *v55; // [rsp+18h] [rbp-108h]
  int v56; // [rsp+20h] [rbp-100h]
  __int64 v57; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+20h] [rbp-100h]
  __int64 *v59; // [rsp+28h] [rbp-F8h]
  _QWORD *v61; // [rsp+38h] [rbp-E8h]
  __int64 v62; // [rsp+48h] [rbp-D8h] BYREF
  void *dest; // [rsp+50h] [rbp-D0h]
  size_t v64; // [rsp+58h] [rbp-C8h]
  _QWORD v65[2]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD *v66; // [rsp+70h] [rbp-B0h] BYREF
  size_t v67; // [rsp+78h] [rbp-A8h]
  _QWORD v68[2]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD v69[4]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v70; // [rsp+B0h] [rbp-70h]
  _QWORD *v71; // [rsp+C0h] [rbp-60h] BYREF
  size_t v72; // [rsp+C8h] [rbp-58h]
  _QWORD v73[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int16 v74; // [rsp+E0h] [rbp-40h]

  result = &a1[a2];
  v62 = a6;
  v55 = result;
  v59 = a1;
  if ( result != a1 )
  {
    while ( 1 )
    {
      v8 = *v59;
      v9 = *(_BYTE *)(*v59 - 16);
      if ( (v9 & 2) != 0 )
      {
        v10 = *(_QWORD **)(v8 - 32);
        v11 = *(unsigned int *)(v8 - 24);
      }
      else
      {
        v11 = (*(_WORD *)(v8 - 16) >> 6) & 0xF;
        v10 = (_QWORD *)(v8 - 8LL * ((v9 >> 2) & 0xF) - 16);
      }
      v61 = &v10[v11];
      if ( v10 != v61 )
        break;
LABEL_29:
      result = ++v59;
      if ( v55 == v59 )
        return result;
    }
    while ( 1 )
    {
      v12 = (_BYTE *)*v10;
      if ( (unsigned __int8)(*(_BYTE *)*v10 - 5) <= 0x1Fu )
        break;
LABEL_28:
      if ( v61 == ++v10 )
        goto LABEL_29;
    }
    LOBYTE(v65[0]) = 0;
    dest = v65;
    v64 = 0;
    v13 = *(v12 - 16);
    if ( (v13 & 2) != 0 )
    {
      if ( *((_DWORD *)v12 - 6) <= 2u )
        goto LABEL_8;
      v28 = *((_QWORD *)v12 - 4);
    }
    else
    {
      if ( ((*((_WORD *)v12 - 8) >> 6) & 0xFu) <= 2 )
        goto LABEL_8;
      v28 = (__int64)&v12[-8 * ((v13 >> 2) & 0xF) - 16];
    }
    v29 = *(_BYTE **)(v28 + 16);
    if ( !v29 || *v29 || (v30 = sub_B91420((__int64)v29), !v31) )
    {
LABEL_8:
      v71 = v73;
      if ( &a4[a5] && !a4 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v69[0] = a5;
      if ( a5 > 0xF )
      {
        v71 = (_QWORD *)sub_22409D0(&v71, v69, 0);
        v34 = v71;
        v73[0] = v69[0];
      }
      else
      {
        if ( a5 == 1 )
        {
          v14 = v73;
          LOBYTE(v73[0]) = *a4;
          v15 = 1;
          goto LABEL_13;
        }
        if ( !a5 )
        {
          v15 = 0;
          v14 = v73;
          goto LABEL_13;
        }
        v34 = v73;
      }
      memcpy(v34, a4, a5);
      v15 = v69[0];
      v14 = v71;
LABEL_13:
      v72 = v15;
      *((_BYTE *)v14 + v15) = 0;
      v16 = dest;
      if ( v71 == v73 )
      {
        v35 = v72;
        if ( v72 )
        {
          if ( v72 == 1 )
            *(_BYTE *)dest = v73[0];
          else
            memcpy(dest, v73, v72);
          v35 = v72;
          v16 = dest;
        }
        v64 = v35;
        v16[v35] = 0;
        v16 = v71;
        goto LABEL_17;
      }
      if ( dest == v65 )
      {
        dest = v71;
        v64 = v72;
        v65[0] = v73[0];
      }
      else
      {
        v17 = v65[0];
        dest = v71;
        v64 = v72;
        v65[0] = v73[0];
        if ( v16 )
        {
          v71 = v16;
          v73[0] = v17;
          goto LABEL_17;
        }
      }
      v71 = v73;
      v16 = v73;
LABEL_17:
      v72 = 0;
      *v16 = 0;
      if ( v71 != v73 )
        j_j___libc_free_0(v71, v73[0] + 1LL);
      goto LABEL_19;
    }
    v69[0] = v30;
    v69[2] = ":";
    v70 = 773;
    v71 = v69;
    v69[1] = v31;
    v73[0] = a4;
    v73[1] = a5;
    v74 = 1282;
    sub_CA0F50((__int64 *)&v66, (void **)&v71);
    v32 = dest;
    if ( v66 == v68 )
    {
      v36 = v67;
      if ( v67 )
      {
        if ( v67 == 1 )
          *(_BYTE *)dest = v68[0];
        else
          memcpy(dest, v68, v67);
        v36 = v67;
        v32 = dest;
      }
      v64 = v36;
      *((_BYTE *)v32 + v36) = 0;
      v32 = v66;
      goto LABEL_43;
    }
    if ( dest == v65 )
    {
      dest = v66;
      v64 = v67;
      v65[0] = v68[0];
    }
    else
    {
      v33 = v65[0];
      dest = v66;
      v64 = v67;
      v65[0] = v68[0];
      if ( v32 )
      {
        v66 = v32;
        v68[0] = v33;
        goto LABEL_43;
      }
    }
    v66 = v68;
    v32 = v68;
LABEL_43:
    v67 = 0;
    *(_BYTE *)v32 = 0;
    if ( v66 != v68 )
      j_j___libc_free_0(v66, v68[0] + 1LL);
LABEL_19:
    v18 = *(v12 - 16);
    if ( (v18 & 2) != 0 )
    {
      if ( *((_DWORD *)v12 - 6) <= 1u )
        goto LABEL_31;
      v19 = *((_QWORD *)v12 - 4);
    }
    else
    {
      if ( ((*((_WORD *)v12 - 8) >> 6) & 0xFu) <= 1 )
        goto LABEL_31;
      v19 = (__int64)&v12[-8 * ((v18 >> 2) & 0xF) - 16];
    }
    v20 = *(_BYTE **)(v19 + 8);
    if ( v20 && (unsigned __int8)(*v20 - 5) <= 0x1Fu )
    {
LABEL_24:
      v21 = sub_B8CD90(&v62, (__int64)dest, v64, (__int64)v20);
      v22 = *(_DWORD *)(a3 + 24);
      v23 = v21;
      if ( v22 )
      {
        v24 = *(_QWORD *)(a3 + 8);
        v25 = (v22 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v26 = (_QWORD *)(v24 + 16LL * v25);
        v27 = (_BYTE *)*v26;
        if ( v12 == (_BYTE *)*v26 )
        {
LABEL_26:
          if ( dest != v65 )
            j_j___libc_free_0(dest, v65[0] + 1LL);
          goto LABEL_28;
        }
        v56 = 1;
        v37 = 0;
        while ( v27 != (_BYTE *)-4096LL )
        {
          if ( v37 || v27 != (_BYTE *)-8192LL )
            v26 = v37;
          v25 = (v22 - 1) & (v56 + v25);
          v27 = *(_BYTE **)(v24 + 16LL * v25);
          if ( v12 == v27 )
            goto LABEL_26;
          ++v56;
          v37 = v26;
          v26 = (_QWORD *)(v24 + 16LL * v25);
        }
        if ( !v37 )
          v37 = v26;
        v38 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v39 = v38 + 1;
        if ( 4 * v39 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a3 + 20) - v39 > v22 >> 3 )
            goto LABEL_71;
          v53 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
          v58 = v23;
          sub_AEABB0(a3, v22);
          v47 = *(_DWORD *)(a3 + 24);
          if ( !v47 )
          {
LABEL_104:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v48 = v47 - 1;
          v49 = *(_QWORD *)(a3 + 8);
          v46 = 0;
          v23 = v58;
          v50 = 1;
          v39 = *(_DWORD *)(a3 + 16) + 1;
          v51 = v48 & v53;
          v37 = (_QWORD *)(v49 + 16LL * (v48 & v53));
          v52 = *v37;
          if ( v12 == (_BYTE *)*v37 )
            goto LABEL_71;
          while ( v52 != -4096 )
          {
            if ( v52 == -8192 && !v46 )
              v46 = v37;
            v51 = v48 & (v50 + v51);
            v37 = (_QWORD *)(v49 + 16LL * v51);
            v52 = *v37;
            if ( v12 == (_BYTE *)*v37 )
              goto LABEL_71;
            ++v50;
          }
          goto LABEL_79;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      v57 = v23;
      sub_AEABB0(a3, 2 * v22);
      v40 = *(_DWORD *)(a3 + 24);
      if ( !v40 )
        goto LABEL_104;
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a3 + 8);
      v23 = v57;
      v43 = v41 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v39 = *(_DWORD *)(a3 + 16) + 1;
      v37 = (_QWORD *)(v42 + 16LL * v43);
      v44 = *v37;
      if ( v12 == (_BYTE *)*v37 )
        goto LABEL_71;
      v45 = 1;
      v46 = 0;
      while ( v44 != -4096 )
      {
        if ( v44 == -8192 && !v46 )
          v46 = v37;
        v43 = v41 & (v45 + v43);
        v37 = (_QWORD *)(v42 + 16LL * v43);
        v44 = *v37;
        if ( v12 == (_BYTE *)*v37 )
          goto LABEL_71;
        ++v45;
      }
LABEL_79:
      if ( v46 )
        v37 = v46;
LABEL_71:
      *(_DWORD *)(a3 + 16) = v39;
      if ( *v37 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v37 = v12;
      v37[1] = v23;
      goto LABEL_26;
    }
LABEL_31:
    v20 = 0;
    goto LABEL_24;
  }
  return result;
}
