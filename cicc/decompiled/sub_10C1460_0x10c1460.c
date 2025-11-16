// Function: sub_10C1460
// Address: 0x10c1460
//
__int64 __fastcall sub_10C1460(__int64 a1, _DWORD **a2, unsigned __int8 *a3, unsigned int a4)
{
  unsigned __int8 v8; // al
  char v10; // al
  _BYTE *v11; // rsi
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rax
  _BYTE *v24; // rax
  unsigned int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned int v28; // ebx
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r14
  int v35; // r13d
  int v36; // eax
  int v37; // ebx
  __int64 v38; // rax
  _BYTE *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r15
  __int64 v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // rax
  unsigned __int64 **v45; // r15
  unsigned __int64 v46; // rax
  unsigned __int64 *v47; // rax
  int v48; // eax
  int v51; // eax
  int v52; // [rsp+0h] [rbp-80h]
  int v53; // [rsp+4h] [rbp-7Ch]
  int v54; // [rsp+4h] [rbp-7Ch]
  int v55; // [rsp+8h] [rbp-78h]
  _BYTE *v56; // [rsp+8h] [rbp-78h]
  __int64 v57; // [rsp+8h] [rbp-78h]
  __int64 v58; // [rsp+10h] [rbp-70h] BYREF
  __int64 v59; // [rsp+18h] [rbp-68h] BYREF
  __int64 v60; // [rsp+20h] [rbp-60h] BYREF
  __int64 v61; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v62; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v63; // [rsp+38h] [rbp-48h] BYREF
  __int64 *v64; // [rsp+40h] [rbp-40h]

  v8 = *a3;
  if ( **a2 == 33 )
  {
    if ( v8 <= 0x1Cu )
      goto LABEL_4;
    if ( v8 != 67 )
      goto LABEL_3;
    v13 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( *v13 != 59 )
      goto LABEL_4;
    if ( !*((_QWORD *)v13 - 8) )
      goto LABEL_4;
    v58 = *((_QWORD *)v13 - 8);
    v14 = *((_QWORD *)v13 - 4);
    if ( !v14 )
      goto LABEL_4;
    v59 = v14;
LABEL_16:
    if ( a4 )
      v15 = v59;
    else
      v15 = v58;
    *(_QWORD *)a1 = v15;
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  v62 = 0;
  v63 = &v58;
  v64 = &v59;
  if ( v8 != 59 )
    goto LABEL_3;
  v10 = sub_995B10(&v62, *((_QWORD *)a3 - 8));
  v11 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( v10 )
  {
    if ( *v11 == 67 )
    {
      v16 = (_BYTE *)*((_QWORD *)v11 - 4);
      if ( *v16 == 59 )
      {
        v17 = *((_QWORD *)v16 - 8);
        if ( v17 )
        {
          *v63 = v17;
          v18 = *((_QWORD *)v16 - 4);
          if ( v18 )
          {
LABEL_23:
            *v64 = v18;
            goto LABEL_16;
          }
          v11 = (_BYTE *)*((_QWORD *)a3 - 4);
        }
      }
    }
  }
  if ( (unsigned __int8)sub_995B10(&v62, (__int64)v11) )
  {
    v12 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v12 == 67 )
    {
      v39 = (_BYTE *)*((_QWORD *)v12 - 4);
      if ( *v39 == 59 )
      {
        v40 = *((_QWORD *)v39 - 8);
        if ( v40 )
        {
          *v63 = v40;
          v18 = *((_QWORD *)v39 - 4);
          if ( v18 )
            goto LABEL_23;
        }
      }
    }
  }
  v8 = *a3;
LABEL_3:
  if ( v8 != 82 )
  {
LABEL_4:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v19 = (unsigned int)**a2;
  v20 = *((_WORD *)a3 + 1) & 0x3F;
  if ( (_DWORD)v19 == v20 )
  {
    v32 = *(_QWORD *)&a3[32 * a4 - 64];
    v33 = *(_QWORD *)(v32 + 16);
    if ( !v33 )
      goto LABEL_4;
    if ( *(_QWORD *)(v33 + 8) )
      goto LABEL_4;
    if ( *(_BYTE *)v32 != 67 )
      goto LABEL_4;
    v34 = *(_QWORD *)(v32 - 32);
    if ( !v34 )
      goto LABEL_4;
    v35 = sub_BCB060(*(_QWORD *)(v34 + 8));
    v36 = sub_BCB060(*(_QWORD *)(v32 + 8));
    LOBYTE(v64) = 0;
    v37 = v36;
    v62 = &v60;
    v63 = &v61;
    v38 = *(_QWORD *)(v34 + 16);
    if ( v38 )
    {
      if ( !*(_QWORD *)(v38 + 8) && *(_BYTE *)v34 == 55 )
      {
        if ( *(_QWORD *)(v34 - 64) )
        {
          v60 = *(_QWORD *)(v34 - 64);
          if ( (unsigned __int8)sub_991580((__int64)&v63, *(_QWORD *)(v34 - 32)) )
          {
            v45 = (unsigned __int64 **)v61;
            if ( *(_DWORD *)(v61 + 8) > 0x40u )
            {
              v55 = *(_DWORD *)(v61 + 8);
              if ( v55 - (unsigned int)sub_C444A0(v61) > 0x40 )
                goto LABEL_52;
              v46 = **v45;
            }
            else
            {
              v46 = *(_QWORD *)v61;
            }
            if ( (unsigned int)(v35 - v37) >= v46 )
            {
              if ( *((_DWORD *)v45 + 2) <= 0x40u )
                v47 = *v45;
              else
                v47 = (unsigned __int64 *)**v45;
              *(_QWORD *)a1 = v60;
              *(_DWORD *)(a1 + 8) = (_DWORD)v47;
              *(_DWORD *)(a1 + 12) = v37;
              *(_BYTE *)(a1 + 16) = 1;
              return a1;
            }
          }
        }
      }
    }
LABEL_52:
    *(_QWORD *)a1 = v34;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 12) = v37;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  if ( (_DWORD)v19 == 32 && v20 == 36 )
  {
    v41 = *((_QWORD *)a3 - 4);
    if ( *(_BYTE *)v41 == 17 )
    {
      v22 = v41 + 24;
      if ( *(_DWORD *)(v41 + 32) > 0x40u )
      {
        v48 = sub_C44630(v22);
        v22 = v41 + 24;
        if ( v48 == 1 )
          goto LABEL_39;
      }
      else
      {
        v42 = *(_QWORD *)(v41 + 24);
        if ( v42 )
        {
          v19 = v42 - 1;
          if ( (v42 & (v42 - 1)) == 0 )
            goto LABEL_39;
        }
      }
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v41 + 8) + 8LL) - 17 > 1 )
        goto LABEL_4;
    }
    else
    {
      v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v41 + 8) + 8LL) - 17;
      if ( (unsigned int)v19 > 1 || *(_BYTE *)v41 > 0x15u )
        goto LABEL_4;
    }
    v43 = sub_AD7630(v41, 1, v19);
    if ( !v43 || *v43 != 17 )
      goto LABEL_4;
    v22 = (__int64)(v43 + 24);
    if ( *((_DWORD *)v43 + 8) > 0x40u )
    {
      v56 = v43 + 24;
      v51 = sub_C44630(v22);
      v22 = (__int64)v56;
      if ( v51 != 1 )
        goto LABEL_4;
    }
    else
    {
      v44 = *((_QWORD *)v43 + 3);
      if ( !v44 || (v44 & (v44 - 1)) != 0 )
        goto LABEL_4;
    }
    goto LABEL_39;
  }
  if ( v20 != 34 || (_DWORD)v19 != 33 )
    goto LABEL_4;
  v21 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v21 != 17 )
  {
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 || *(_BYTE *)v21 > 0x15u )
      goto LABEL_4;
LABEL_34:
    v24 = sub_AD7630(v21, 1, v19);
    if ( !v24 || *v24 != 17 )
      goto LABEL_4;
    v25 = *((_DWORD *)v24 + 8);
    v22 = (__int64)(v24 + 24);
    if ( v25 > 0x40 )
    {
      v57 = (__int64)(v24 + 24);
      v54 = sub_C445E0(v22);
      if ( !v54 )
        goto LABEL_4;
      v22 = v57;
      if ( v25 != (unsigned int)sub_C444A0(v57) + v54 )
        goto LABEL_4;
    }
    else
    {
      v26 = *((_QWORD *)v24 + 3);
      if ( !v26 || (v26 & (v26 + 1)) != 0 )
        goto LABEL_4;
    }
    goto LABEL_39;
  }
  v22 = v21 + 24;
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
  {
    v52 = *(_DWORD *)(v21 + 32);
    v53 = sub_C445E0(v22);
    if ( !v53 || (v22 = v21 + 24, v19 = (unsigned int)sub_C444A0(v21 + 24) + v53, v52 != (_DWORD)v19) )
    {
LABEL_33:
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + 8LL) - 17 > 1 )
        goto LABEL_4;
      goto LABEL_34;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v21 + 24);
    if ( !v23 )
      goto LABEL_33;
    v19 = v23 + 1;
    if ( (v23 & (v23 + 1)) != 0 )
      goto LABEL_33;
  }
LABEL_39:
  v27 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v27 != 59 )
    goto LABEL_4;
  v28 = *(_DWORD *)(v22 + 8);
  if ( **a2 == 33 )
  {
    if ( v28 > 0x40 )
      v29 = sub_C44630(v22);
    else
      v29 = sub_39FAC40(*(_QWORD *)v22);
  }
  else if ( v28 <= 0x40 )
  {
    _RAX = *(_QWORD *)v22;
    __asm { tzcnt   rdx, rax }
    v29 = 64;
    if ( *(_QWORD *)v22 )
      v29 = _RDX;
    if ( v29 > v28 )
      v29 = *(_DWORD *)(v22 + 8);
  }
  else
  {
    v29 = sub_C44590(v22);
  }
  if ( (*(_BYTE *)(v27 + 7) & 0x40) != 0 )
    v30 = *(_QWORD *)(v27 - 8);
  else
    v30 = v27 - 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF);
  *(_DWORD *)(a1 + 8) = v29;
  v31 = *(_QWORD *)(v30 + 32LL * a4);
  *(_DWORD *)(a1 + 12) = v28 - v29;
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v31;
  return a1;
}
