// Function: sub_1118A30
// Address: 0x1118a30
//
_QWORD *__fastcall sub_1118A30(__int64 a1, __int64 a2, __int64 a3, __int64 **a4)
{
  __int64 v4; // rax
  __int64 v7; // rcx
  unsigned int v8; // r15d
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // r9
  __int16 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r13
  _QWORD *result; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // eax
  bool v20; // al
  bool v21; // al
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r13
  _QWORD *v29; // rax
  __int64 v30; // rdx
  bool v31; // al
  char v32; // r8
  __int64 v33; // r15
  __int64 v34; // r13
  int v35; // eax
  bool v36; // al
  int v37; // eax
  unsigned int **v38; // rdi
  __int64 v39; // rax
  _BYTE *v40; // rsi
  __int64 v41; // r15
  __int64 v42; // r13
  _QWORD *v43; // rdx
  bool v44; // cf
  int v45; // eax
  unsigned int v46; // r10d
  unsigned int v47; // edx
  int v48; // eax
  __int64 *v49; // rdi
  __int64 v50; // rbx
  __int64 v51; // r13
  _QWORD *v52; // rax
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // [rsp+8h] [rbp-C8h]
  int v58; // [rsp+10h] [rbp-C0h]
  int v59; // [rsp+10h] [rbp-C0h]
  unsigned int v60; // [rsp+10h] [rbp-C0h]
  unsigned int v61; // [rsp+18h] [rbp-B8h]
  __int16 v62; // [rsp+18h] [rbp-B8h]
  int v63; // [rsp+18h] [rbp-B8h]
  _QWORD *v64; // [rsp+18h] [rbp-B8h]
  _QWORD *v65; // [rsp+18h] [rbp-B8h]
  _QWORD *v66; // [rsp+18h] [rbp-B8h]
  __int64 v67; // [rsp+18h] [rbp-B8h]
  _QWORD *v68; // [rsp+18h] [rbp-B8h]
  int v69; // [rsp+18h] [rbp-B8h]
  int v70; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v71; // [rsp+18h] [rbp-B8h]
  unsigned int v72; // [rsp+18h] [rbp-B8h]
  _QWORD *v73; // [rsp+18h] [rbp-B8h]
  _QWORD *v74; // [rsp+18h] [rbp-B8h]
  _QWORD *v75; // [rsp+18h] [rbp-B8h]
  _QWORD *v76; // [rsp+18h] [rbp-B8h]
  __int64 v77[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v78[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v80; // [rsp+48h] [rbp-88h]
  __int16 v81; // [rsp+60h] [rbp-70h]
  __int64 *v82; // [rsp+70h] [rbp-60h] BYREF
  char v83; // [rsp+78h] [rbp-58h]
  __int16 v84; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a3 - 32);
  if ( !v4 || *(_BYTE *)v4 || (v7 = *(_QWORD *)(a3 + 80), *(_QWORD *)(v4 + 24) != v7) )
    BUG();
  v8 = *(_DWORD *)(v4 + 36);
  v9 = a1;
  v10 = *(_QWORD *)(a3 + 8);
  v11 = *((unsigned int *)a4 + 2);
  v12 = *(_WORD *)(a2 + 2) & 0x3F;
  v13 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v8 == 67 )
    goto LABEL_63;
  if ( v8 <= 0x43 )
  {
    if ( v8 == 15 )
    {
      v24 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      sub_C496B0((__int64)&v79, (__int64)a4);
    }
    else
    {
      if ( v8 > 0xF )
      {
        if ( v8 != 65 )
        {
          if ( v8 != 66 )
            return 0;
          if ( (unsigned int)v11 <= 0x40 )
          {
            if ( !*a4 )
              goto LABEL_12;
            v53 = (__int64)*a4;
            goto LABEL_80;
          }
          v61 = *((_DWORD *)a4 + 2);
          if ( v61 == (unsigned int)sub_C444A0((__int64)a4) )
          {
LABEL_12:
            v14 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
            v15 = sub_AD6530(v10, a2);
            goto LABEL_13;
          }
          v54 = sub_C444A0((__int64)a4);
          v11 = v61;
          if ( v61 - v54 <= 0x40 )
          {
            v53 = **a4;
LABEL_80:
            if ( v11 == v53 )
            {
              v14 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
              v15 = sub_AD62B0(v10);
LABEL_13:
              v84 = 257;
              result = sub_BD2C40(72, unk_3F10FD0);
              if ( result )
              {
                v17 = v15;
                v18 = v14;
LABEL_33:
                v64 = result;
                sub_1113300((__int64)result, v12, v18, v17, (__int64)&v82);
                return v64;
              }
              return result;
            }
          }
          return 0;
        }
LABEL_63:
        if ( (unsigned int)v11 > 0x40 )
        {
          v59 = *((_DWORD *)a4 + 2);
          v71 = (unsigned int)v11;
          v45 = sub_C444A0((__int64)a4);
          LODWORD(v11) = v59;
          v9 = a1;
          if ( (unsigned int)(v59 - v45) > 0x40 )
            return 0;
          v43 = (_QWORD *)**a4;
          v44 = v71 < (unsigned __int64)v43;
          if ( (_QWORD *)v71 == v43 )
            goto LABEL_12;
        }
        else
        {
          v43 = *a4;
          v44 = (unsigned int)v11 < (unsigned __int64)*a4;
          if ( (__int64 *)(unsigned int)v11 == *a4 )
            goto LABEL_12;
        }
        result = 0;
        if ( v44 )
          return result;
        v46 = (unsigned int)v43;
        if ( (_DWORD)v11 != (_DWORD)v43 )
        {
          result = *(_QWORD **)(a3 + 16);
          if ( !result )
            return result;
          if ( !result[1] )
          {
            v47 = (_DWORD)v43 + 1;
            v57 = v9;
            v60 = v46;
            v72 = v11;
            if ( v8 == 67 )
            {
              sub_F0A5D0((__int64)v77, v11, v47);
              sub_9866F0((__int64)v78, v72, v60);
            }
            else
            {
              sub_109DDE0((__int64)v77, v11, v47);
              sub_9866F0((__int64)v78, v72, v72 - 1 - v60);
            }
            v48 = *(_DWORD *)(a3 + 4);
            v49 = *(__int64 **)(v57 + 32);
            v81 = 257;
            v50 = sub_10BC480(v49, *(_QWORD *)(a3 - 32LL * (v48 & 0x7FFFFFF)), (__int64)v77, (__int64)&v79);
            v51 = sub_AD8D80(v10, (__int64)v78);
            v84 = 257;
            v52 = sub_BD2C40(72, unk_3F10FD0);
            if ( v52 )
            {
              v73 = v52;
              sub_1113300((__int64)v52, v12, v50, v51, (__int64)&v82);
              v52 = v73;
            }
            v74 = v52;
            sub_969240(v78);
            sub_969240(v77);
            return v74;
          }
        }
        return 0;
      }
      if ( v8 == 1 )
      {
        if ( (unsigned int)v11 <= 0x40 )
        {
          v31 = *a4 == 0;
        }
        else
        {
          v69 = *((_DWORD *)a4 + 2);
          v31 = v69 == (unsigned int)sub_C444A0((__int64)a4);
        }
        if ( v31 || (v32 = sub_986B30((__int64 *)a4, a2, v13, v7, v9), result = 0, v32) )
        {
          v33 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
          v84 = 257;
          v34 = sub_AD8D80(v10, (__int64)a4);
          result = sub_BD2C40(72, unk_3F10FD0);
          if ( result )
          {
            v17 = v34;
            v18 = v33;
            goto LABEL_33;
          }
        }
        return result;
      }
      if ( v8 != 14 )
        return 0;
      v24 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      sub_C48440((__int64)&v79, (unsigned __int8 *)a4);
    }
    v25 = sub_AD8D80(v10, (__int64)&v79);
    v84 = 257;
    result = sub_BD2C40(72, unk_3F10FD0);
    if ( result )
    {
      v65 = result;
      sub_1113300((__int64)result, v12, v24, v25, (__int64)&v82);
      result = v65;
    }
    if ( v80 > 0x40 && v79 )
    {
      v66 = result;
      j_j___libc_free_0_0(v79);
      return v66;
    }
    return result;
  }
  if ( v8 == 359 )
    goto LABEL_56;
  if ( v8 > 0x167 )
  {
    if ( v8 != 365 )
    {
      if ( v8 != 371 )
        return 0;
      if ( (unsigned int)v11 <= 0x40 )
      {
        v20 = *a4 == 0;
      }
      else
      {
        v58 = *((_DWORD *)a4 + 2);
        v62 = *(_WORD *)(a2 + 2) & 0x3F;
        v19 = sub_C444A0((__int64)a4);
        LOWORD(v13) = v62;
        v20 = v58 == v19;
      }
      if ( !v20 )
        return 0;
      v12 = 3 * ((_WORD)v13 == 32) + 34;
      goto LABEL_31;
    }
LABEL_56:
    if ( (unsigned int)v11 <= 0x40 )
    {
      v36 = *a4 == 0;
    }
    else
    {
      v70 = *((_DWORD *)a4 + 2);
      v35 = sub_C444A0((__int64)a4);
      v9 = a1;
      v36 = v70 == v35;
    }
    if ( v36 )
    {
      result = *(_QWORD **)(a3 + 16);
      if ( !result )
        return result;
      if ( !result[1] )
      {
        v37 = *(_DWORD *)(a3 + 4);
        v38 = *(unsigned int ***)(v9 + 32);
        v84 = 257;
        v39 = v37 & 0x7FFFFFF;
        v40 = *(_BYTE **)(a3 - 32 * v39);
        v41 = sub_A82480(v38, v40, *(_BYTE **)(a3 + 32 * (1 - v39)), (__int64)&v82);
        v42 = sub_AD6530(v10, (__int64)v40);
        v84 = 257;
        result = sub_BD2C40(72, unk_3F10FD0);
        if ( result )
        {
          v17 = v42;
          v18 = v41;
          goto LABEL_33;
        }
        return result;
      }
    }
    return 0;
  }
  if ( v8 > 0xB5 )
  {
    if ( v8 != 338 )
      return 0;
    if ( (unsigned int)v11 <= 0x40 )
    {
      v21 = *a4 == 0;
    }
    else
    {
      v63 = *((_DWORD *)a4 + 2);
      v21 = v63 == (unsigned int)sub_C444A0((__int64)a4);
    }
    if ( !v21 )
      return 0;
LABEL_31:
    v22 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v23 = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
    v84 = 257;
    result = sub_BD2C40(72, unk_3F10FD0);
    if ( result )
    {
      v17 = v23;
      v18 = v22;
      goto LABEL_33;
    }
    return result;
  }
  if ( v8 <= 0xB3 )
    return 0;
  v26 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  if ( *(_QWORD *)(a3 + 32 * (1 - v26)) != *(_QWORD *)(a3 - 32 * v26) )
    return 0;
  v83 = 0;
  v82 = v78;
  if ( !(unsigned __int8)sub_991580((__int64)&v82, *(_QWORD *)(a3 + 32 * (2 - v26))) )
    return 0;
  v27 = *(_QWORD *)(a3 - 32);
  if ( !v27 || *(_BYTE *)v27 || *(_QWORD *)(v27 + 24) != *(_QWORD *)(a3 + 80) )
    BUG();
  v67 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  if ( *(_DWORD *)(v27 + 36) == 180 )
  {
    sub_C4B870((__int64)&v79, (__int64)a4, v78[0]);
    v55 = sub_AD8D80(v10, (__int64)&v79);
    v84 = 257;
    v29 = sub_BD2C40(72, unk_3F10FD0);
    if ( v29 )
    {
      v56 = v67;
      v75 = v29;
      sub_1113300((__int64)v29, v12, v56, v55, (__int64)&v82);
      v29 = v75;
    }
  }
  else
  {
    sub_C4B840((__int64)&v79, (__int64)a4, v78[0]);
    v28 = sub_AD8D80(v10, (__int64)&v79);
    v84 = 257;
    v29 = sub_BD2C40(72, unk_3F10FD0);
    if ( v29 )
    {
      v30 = v67;
      v68 = v29;
      sub_1113300((__int64)v29, v12, v30, v28, (__int64)&v82);
      sub_969240(&v79);
      return v68;
    }
  }
  v76 = v29;
  sub_969240(&v79);
  return v76;
}
