// Function: sub_C76560
// Address: 0xc76560
//
_QWORD *__fastcall sub_C76560(_QWORD *a1, __int64 a2, __int64 a3, char a4, char a5)
{
  unsigned int v8; // r12d
  unsigned int v9; // r15d
  bool v10; // al
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned int *v13; // rdx
  unsigned int v14; // r12d
  unsigned int v15; // eax
  unsigned int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // edx
  bool v26; // al
  unsigned int v29; // edx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned int *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  unsigned int v41; // edx
  __int64 v42; // rcx
  bool v43; // cc
  __int64 v44; // rdi
  int v45; // eax
  unsigned int v46; // ecx
  int v47; // eax
  bool v48; // al
  unsigned int v49; // esi
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // eax
  __int64 v55; // rsi
  unsigned __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rsi
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  unsigned int v62; // eax
  unsigned int v63; // [rsp+4h] [rbp-BCh]
  unsigned int v64; // [rsp+4h] [rbp-BCh]
  __int64 v65; // [rsp+8h] [rbp-B8h]
  __int64 v66; // [rsp+8h] [rbp-B8h]
  unsigned int v67; // [rsp+10h] [rbp-B0h]
  unsigned int v68; // [rsp+10h] [rbp-B0h]
  unsigned int v69; // [rsp+10h] [rbp-B0h]
  unsigned int v70; // [rsp+10h] [rbp-B0h]
  unsigned int v71; // [rsp+10h] [rbp-B0h]
  unsigned int v72; // [rsp+18h] [rbp-A8h]
  int v75; // [rsp+28h] [rbp-98h]
  int v76; // [rsp+28h] [rbp-98h]
  __int64 v77; // [rsp+28h] [rbp-98h]
  unsigned int v78; // [rsp+28h] [rbp-98h]
  unsigned int v79; // [rsp+28h] [rbp-98h]
  const void **v80; // [rsp+30h] [rbp-90h]
  const void **v81; // [rsp+38h] [rbp-88h]
  unsigned int v82; // [rsp+38h] [rbp-88h]
  unsigned int v83; // [rsp+38h] [rbp-88h]
  unsigned int *v84; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v85; // [rsp+48h] [rbp-78h]
  __int64 v86; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v87; // [rsp+58h] [rbp-68h]
  __int64 v88; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v89; // [rsp+68h] [rbp-58h]
  unsigned __int64 v90; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v91; // [rsp+78h] [rbp-48h]
  unsigned __int64 v92; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v93; // [rsp+88h] [rbp-38h]

  v8 = *(_DWORD *)(a2 + 8);
  *((_DWORD *)a1 + 2) = v8;
  v80 = (const void **)(a1 + 2);
  if ( v8 > 0x40 )
  {
    sub_C43690((__int64)a1, 0, 0);
    *((_DWORD *)a1 + 6) = v8;
    sub_C43690((__int64)v80, 0, 0);
  }
  else
  {
    *a1 = 0;
    *((_DWORD *)a1 + 6) = v8;
    a1[2] = 0;
  }
  v81 = (const void **)(a3 + 16);
  v91 = *(_DWORD *)(a3 + 24);
  if ( v91 > 0x40 )
  {
    sub_C43780((__int64)&v90, v81);
    v72 = v91;
    if ( v91 <= 0x40 )
      goto LABEL_5;
    if ( v72 - (unsigned int)sub_C444A0((__int64)&v90) > 0x40 )
    {
      if ( !v90 )
        goto LABEL_6;
    }
    else if ( *(_QWORD *)v90 <= (unsigned __int64)v8 )
    {
      v9 = *(_DWORD *)v90;
LABEL_50:
      j_j___libc_free_0_0(v90);
      goto LABEL_7;
    }
    v9 = v8;
    goto LABEL_50;
  }
  v90 = *(_QWORD *)(a3 + 16);
LABEL_5:
  if ( v8 < v90 )
  {
LABEL_6:
    v9 = v8;
    goto LABEL_7;
  }
  v9 = v90;
LABEL_7:
  if ( !v9 )
    v9 = a4 != 0;
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    v10 = *(_QWORD *)a2 == 0;
  }
  else
  {
    v75 = *(_DWORD *)(a2 + 8);
    v10 = v75 == (unsigned int)sub_C444A0(a2);
  }
  if ( v10 )
  {
    if ( *(_DWORD *)(a2 + 24) <= 0x40u )
    {
      v26 = *(_QWORD *)(a2 + 16) == 0;
    }
    else
    {
      v76 = *(_DWORD *)(a2 + 24);
      v26 = v76 == (unsigned int)sub_C444A0(a2 + 16);
    }
    if ( v26 )
    {
      if ( v9 != v8 )
        return a1;
      v50 = *((unsigned int *)a1 + 2);
      if ( (unsigned int)v50 > 0x40 )
      {
        memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v50 + 63) >> 6));
        v50 = *((unsigned int *)a1 + 2);
        v51 = *a1;
      }
      else
      {
        *a1 = -1;
        v51 = -1;
      }
      v52 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v50;
      if ( (_DWORD)v50 )
      {
        if ( (unsigned int)v50 > 0x40 )
        {
          v53 = (unsigned int)((unsigned __int64)(v50 + 63) >> 6) - 1;
          *(_QWORD *)(v51 + 8 * v53) &= v52;
          goto LABEL_131;
        }
      }
      else
      {
        v52 = 0;
      }
      *a1 = v51 & v52;
LABEL_131:
      v54 = *((_DWORD *)a1 + 6);
      if ( v54 > 0x40 )
        memset((void *)a1[2], 0, 8 * (((unsigned __int64)v54 + 63) >> 6));
      else
        a1[2] = 0;
      return a1;
    }
  }
  v11 = *(_DWORD *)(a3 + 8);
  v91 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v90, (const void **)a3);
    v11 = v91;
    if ( v91 > 0x40 )
    {
      sub_C43D10((__int64)&v90);
      v11 = v91;
      v13 = (unsigned int *)v90;
      goto LABEL_16;
    }
    v12 = v90;
  }
  else
  {
    v12 = *(_QWORD *)a3;
  }
  v13 = (unsigned int *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12);
  if ( !v11 )
    v13 = 0;
LABEL_16:
  v85 = v11;
  v84 = v13;
  v14 = sub_C6EC80((__int64)&v84, v8);
  if ( !a5 )
    goto LABEL_22;
  v15 = *(_DWORD *)(a2 + 24);
  if ( v15 <= 0x40 )
  {
    _RCX = *(_QWORD *)(a2 + 16);
    v29 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v29 = _RSI;
    if ( v15 > v29 )
      v15 = v29;
  }
  else
  {
    v15 = sub_C44590(a2 + 16);
  }
  if ( v9 <= v15 )
  {
    if ( v14 > v15 )
      v14 = v15;
LABEL_22:
    sub_C44AB0((__int64)&v90, a3, 0x20u);
    v16 = v90;
    if ( v91 > 0x40 )
    {
      v16 = *(_DWORD *)v90;
      j_j___libc_free_0_0(v90);
    }
    sub_C44AB0((__int64)&v90, (__int64)v81, 0x20u);
    if ( v91 <= 0x40 )
    {
      v82 = v90;
    }
    else
    {
      v82 = *(_DWORD *)v90;
      j_j___libc_free_0_0(v90);
    }
    v17 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v17 > 0x40 )
    {
      memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v17 + 63) >> 6));
      v17 = *((unsigned int *)a1 + 2);
      v18 = *a1;
    }
    else
    {
      *a1 = -1;
      v18 = -1;
    }
    v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    if ( (_DWORD)v17 )
    {
      if ( (unsigned int)v17 > 0x40 )
      {
        v20 = (unsigned int)((unsigned __int64)(v17 + 63) >> 6) - 1;
        *(_QWORD *)(v18 + 8 * v20) &= v19;
        goto LABEL_31;
      }
    }
    else
    {
      v19 = 0;
    }
    *a1 = v18 & v19;
LABEL_31:
    v21 = *((unsigned int *)a1 + 6);
    if ( (unsigned int)v21 > 0x40 )
    {
      memset((void *)a1[2], -1, 8 * (((unsigned __int64)(unsigned int)v21 + 63) >> 6));
      v21 = *((unsigned int *)a1 + 6);
      v22 = a1[2];
    }
    else
    {
      a1[2] = -1;
      v22 = -1;
    }
    v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
    if ( (_DWORD)v21 )
    {
      if ( (unsigned int)v21 > 0x40 )
      {
        v24 = (unsigned int)((unsigned __int64)(v21 + 63) >> 6) - 1;
        *(_QWORD *)(v22 + 8 * v24) &= v23;
LABEL_36:
        while ( v14 >= v9 )
        {
          if ( (v9 & v16) != 0 || (v9 | v82) != v9 )
            goto LABEL_39;
          v91 = *(_DWORD *)(a2 + 8);
          if ( v91 > 0x40 )
            sub_C43780((__int64)&v90, (const void **)a2);
          else
            v90 = *(_QWORD *)a2;
          v93 = *(_DWORD *)(a2 + 24);
          if ( v93 > 0x40 )
            sub_C43780((__int64)&v92, (const void **)(a2 + 16));
          else
            v92 = *(_QWORD *)(a2 + 16);
          if ( v91 > 0x40 )
          {
            sub_C44B70((__int64)&v90, v9);
          }
          else
          {
            v31 = 0;
            if ( v91 )
              v31 = (__int64)(v90 << (64 - (unsigned __int8)v91)) >> (64 - (unsigned __int8)v91);
            v32 = v31 >> 63;
            v33 = v31 >> v9;
            if ( v9 == v91 )
              v33 = v32;
            v34 = (unsigned int *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v91) & v33);
            if ( !v91 )
              v34 = 0;
            v90 = (unsigned __int64)v34;
          }
          if ( v93 > 0x40 )
          {
            sub_C44B70((__int64)&v92, v9);
            v39 = *((_DWORD *)a1 + 6);
            v89 = v39;
            if ( v39 <= 0x40 )
              goto LABEL_86;
          }
          else
          {
            v35 = 0;
            if ( v93 )
              v35 = (__int64)(v92 << (64 - (unsigned __int8)v93)) >> (64 - (unsigned __int8)v93);
            v36 = v35 >> 63;
            v37 = v35 >> v9;
            if ( v9 == v93 )
              v37 = v36;
            v38 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v93) & v37;
            if ( !v93 )
              v38 = 0;
            v92 = v38;
            v39 = *((_DWORD *)a1 + 6);
            v89 = v39;
            if ( v39 <= 0x40 )
            {
LABEL_86:
              v40 = a1[2];
LABEL_87:
              v77 = v92 & v40;
              v88 = v92 & v40;
              goto LABEL_88;
            }
          }
          sub_C43780((__int64)&v88, v80);
          v39 = v89;
          if ( v89 <= 0x40 )
          {
            v40 = v88;
            goto LABEL_87;
          }
          sub_C43B90(&v88, (__int64 *)&v92);
          v39 = v89;
          v77 = v88;
LABEL_88:
          v41 = *((_DWORD *)a1 + 2);
          v89 = 0;
          v87 = v41;
          if ( v41 > 0x40 )
          {
            v70 = v39;
            sub_C43780((__int64)&v86, (const void **)a1);
            v41 = v87;
            v39 = v70;
            if ( v87 <= 0x40 )
            {
              v49 = v89;
              v42 = v86 & v90;
            }
            else
            {
              sub_C43B90(&v86, (__int64 *)&v90);
              v41 = v87;
              v42 = v86;
              v49 = v89;
              v39 = v70;
            }
            if ( v49 > 0x40 && v88 )
            {
              v64 = v41;
              v66 = v42;
              v71 = v39;
              j_j___libc_free_0_0(v88);
              v41 = v64;
              v42 = v66;
              v39 = v71;
            }
          }
          else
          {
            v42 = *a1 & v90;
          }
          if ( *((_DWORD *)a1 + 2) > 0x40u && *a1 )
          {
            v63 = v41;
            v65 = v42;
            v67 = v39;
            j_j___libc_free_0_0(*a1);
            v41 = v63;
            v42 = v65;
            v39 = v67;
          }
          v43 = *((_DWORD *)a1 + 6) <= 0x40u;
          *a1 = v42;
          *((_DWORD *)a1 + 2) = v41;
          if ( !v43 )
          {
            v44 = a1[2];
            if ( v44 )
            {
              v68 = v39;
              j_j___libc_free_0_0(v44);
              v39 = v68;
            }
          }
          v43 = v93 <= 0x40;
          *((_DWORD *)a1 + 6) = v39;
          a1[2] = v77;
          if ( !v43 && v92 )
            j_j___libc_free_0_0(v92);
          if ( v91 > 0x40 && v90 )
            j_j___libc_free_0_0(v90);
          v25 = *((_DWORD *)a1 + 2);
          if ( v25 <= 0x40 )
          {
            if ( *a1 )
              goto LABEL_39;
            v46 = *((_DWORD *)a1 + 6);
            if ( v46 > 0x40 )
              goto LABEL_105;
          }
          else
          {
            v78 = *((_DWORD *)a1 + 2);
            v45 = sub_C444A0((__int64)a1);
            v25 = v78;
            if ( v78 != v45 )
              goto LABEL_39;
            v46 = *((_DWORD *)a1 + 6);
            if ( v46 > 0x40 )
            {
LABEL_105:
              v69 = v46;
              v79 = v25;
              v47 = sub_C444A0((__int64)v80);
              v25 = v79;
              v48 = v69 == v47;
              goto LABEL_106;
            }
          }
          v48 = a1[2] == 0;
LABEL_106:
          if ( v48 )
            goto LABEL_41;
LABEL_39:
          ++v9;
        }
        v25 = *((_DWORD *)a1 + 2);
LABEL_41:
        if ( v25 <= 0x40 )
        {
          if ( (a1[2] & *a1) == 0 )
            goto LABEL_43;
          *a1 = -1;
          v55 = -1;
        }
        else
        {
          v83 = v25;
          if ( !(unsigned __int8)sub_C446A0(a1, (__int64 *)v80) )
            goto LABEL_43;
          memset((void *)*a1, -1, 8 * (((unsigned __int64)v83 + 63) >> 6));
          v25 = *((_DWORD *)a1 + 2);
          v55 = *a1;
        }
        v56 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v25;
        if ( v25 )
        {
          if ( v25 > 0x40 )
          {
            v57 = (unsigned int)(((unsigned __int64)v25 + 63) >> 6) - 1;
            *(_QWORD *)(v55 + 8 * v57) &= v56;
            goto LABEL_149;
          }
        }
        else
        {
          v56 = 0;
        }
        *a1 = v55 & v56;
        goto LABEL_149;
      }
    }
    else
    {
      v23 = 0;
    }
    a1[2] = v22 & v23;
    goto LABEL_36;
  }
  v58 = *((unsigned int *)a1 + 2);
  if ( (unsigned int)v58 > 0x40 )
  {
    memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v58 + 63) >> 6));
    v58 = *((unsigned int *)a1 + 2);
    v59 = *a1;
  }
  else
  {
    *a1 = -1;
    v59 = -1;
  }
  v60 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v58;
  if ( (_DWORD)v58 )
  {
    if ( (unsigned int)v58 > 0x40 )
    {
      v61 = (unsigned int)((unsigned __int64)(v58 + 63) >> 6) - 1;
      *(_QWORD *)(v59 + 8 * v61) &= v60;
      goto LABEL_149;
    }
  }
  else
  {
    v60 = 0;
  }
  *a1 = v59 & v60;
LABEL_149:
  v62 = *((_DWORD *)a1 + 6);
  if ( v62 > 0x40 )
    memset((void *)a1[2], 0, 8 * (((unsigned __int64)v62 + 63) >> 6));
  else
    a1[2] = 0;
LABEL_43:
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  return a1;
}
