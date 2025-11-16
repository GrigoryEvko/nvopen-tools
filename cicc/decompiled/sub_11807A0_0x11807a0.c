// Function: sub_11807A0
// Address: 0x11807a0
//
__int64 __fastcall sub_11807A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  char v11; // r12
  __int64 result; // rax
  bool v13; // al
  bool v14; // zf
  __int64 v15; // rax
  _BYTE *v16; // rax
  bool v17; // cl
  _BYTE *v18; // rax
  bool v19; // cl
  unsigned int v20; // r15d
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rax
  unsigned int v32; // eax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  unsigned int v35; // eax
  bool v36; // dl
  unsigned int v37; // eax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  unsigned int v41; // eax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  int v46; // [rsp+0h] [rbp-E0h]
  int v47; // [rsp+0h] [rbp-E0h]
  bool v48; // [rsp+4h] [rbp-DCh]
  int v49; // [rsp+4h] [rbp-DCh]
  bool v50; // [rsp+4h] [rbp-DCh]
  int v51; // [rsp+4h] [rbp-DCh]
  int v52; // [rsp+8h] [rbp-D8h]
  int v53; // [rsp+8h] [rbp-D8h]
  __int64 v54; // [rsp+8h] [rbp-D8h]
  int v55; // [rsp+8h] [rbp-D8h]
  __int64 v56; // [rsp+8h] [rbp-D8h]
  bool v57; // [rsp+8h] [rbp-D8h]
  bool v58; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+18h] [rbp-C8h]
  __int64 v61; // [rsp+18h] [rbp-C8h]
  __int64 v62; // [rsp+18h] [rbp-C8h]
  const void **v63; // [rsp+28h] [rbp-B8h] BYREF
  const void ***v64; // [rsp+30h] [rbp-B0h] BYREF
  char v65; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v66; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v67; // [rsp+48h] [rbp-98h]
  unsigned __int64 v68; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v69; // [rsp+58h] [rbp-88h]
  __int16 v70; // [rsp+70h] [rbp-70h]
  __int64 v71; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v72[3]; // [rsp+88h] [rbp-58h] BYREF
  __int16 v73; // [rsp+A0h] [rbp-40h]

  v5 = a2;
  v6 = *(_QWORD *)(a1 - 64);
  v7 = *(_WORD *)(a1 + 2) & 0x3F;
  v60 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)a2 > 0x15u )
    goto LABEL_4;
  if ( sub_AC30F0(a2) )
  {
LABEL_3:
    v7 = sub_B52870(v7);
    v9 = v5;
    v5 = a3;
    a3 = v9;
    goto LABEL_4;
  }
  if ( *(_BYTE *)a2 == 17 )
  {
    if ( *(_DWORD *)(a2 + 32) <= 0x40u )
    {
      v13 = *(_QWORD *)(a2 + 24) == 0;
    }
    else
    {
      v53 = *(_DWORD *)(a2 + 32);
      v13 = v53 == (unsigned int)sub_C444A0(a2 + 24);
    }
  }
  else
  {
    v54 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 > 1 )
      goto LABEL_4;
    v16 = sub_AD7630(a2, 0, v8);
    v17 = 0;
    if ( !v16 || *v16 != 17 )
    {
      if ( *(_BYTE *)(v54 + 8) == 17 )
      {
        v46 = *(_DWORD *)(v54 + 32);
        if ( v46 )
        {
          v42 = 0;
          while ( 1 )
          {
            v48 = v17;
            v43 = sub_AD69F0((unsigned __int8 *)v5, v42);
            if ( !v43 )
              break;
            v17 = v48;
            if ( *(_BYTE *)v43 != 13 )
            {
              if ( *(_BYTE *)v43 != 17 )
                break;
              if ( *(_DWORD *)(v43 + 32) <= 0x40u )
              {
                v17 = *(_QWORD *)(v43 + 24) == 0;
              }
              else
              {
                v49 = *(_DWORD *)(v43 + 32);
                v17 = v49 == (unsigned int)sub_C444A0(v43 + 24);
              }
              if ( !v17 )
                break;
            }
            v42 = (unsigned int)(v42 + 1);
            if ( v46 == (_DWORD)v42 )
            {
              if ( v17 )
                goto LABEL_3;
              goto LABEL_4;
            }
          }
        }
      }
      goto LABEL_4;
    }
    if ( *((_DWORD *)v16 + 8) <= 0x40u )
    {
      v13 = *((_QWORD *)v16 + 3) == 0;
    }
    else
    {
      v55 = *((_DWORD *)v16 + 8);
      v13 = v55 == (unsigned int)sub_C444A0((__int64)(v16 + 24));
    }
  }
  if ( v13 )
    goto LABEL_3;
LABEL_4:
  if ( *(_BYTE *)a3 > 0x15u )
    return 0;
  if ( !sub_AC30F0(a3) )
  {
    if ( *(_BYTE *)a3 == 17 )
    {
      if ( *(_DWORD *)(a3 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(a3 + 24) )
          return 0;
      }
      else
      {
        v52 = *(_DWORD *)(a3 + 32);
        if ( v52 != (unsigned int)sub_C444A0(a3 + 24) )
          return 0;
      }
    }
    else
    {
      v56 = *(_QWORD *)(a3 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v56 + 8) - 17 > 1 )
        return 0;
      v18 = sub_AD7630(a3, 0, v10);
      v19 = 0;
      if ( !v18 || *v18 != 17 )
      {
        if ( *(_BYTE *)(v56 + 8) == 17 )
        {
          v47 = *(_DWORD *)(v56 + 32);
          if ( v47 )
          {
            v44 = 0;
            while ( 1 )
            {
              v50 = v19;
              v45 = sub_AD69F0((unsigned __int8 *)a3, v44);
              if ( !v45 )
                break;
              v19 = v50;
              if ( *(_BYTE *)v45 != 13 )
              {
                if ( *(_BYTE *)v45 != 17 )
                  break;
                if ( *(_DWORD *)(v45 + 32) <= 0x40u )
                {
                  v19 = *(_QWORD *)(v45 + 24) == 0;
                }
                else
                {
                  v51 = *(_DWORD *)(v45 + 32);
                  v19 = v51 == (unsigned int)sub_C444A0(v45 + 24);
                }
                if ( !v19 )
                  break;
              }
              v44 = (unsigned int)(v44 + 1);
              if ( v47 == (_DWORD)v44 )
              {
                if ( v19 )
                  goto LABEL_6;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v20 = *((_DWORD *)v18 + 8);
      if ( !(v20 <= 0x40 ? *((_QWORD *)v18 + 3) == 0 : v20 == (unsigned int)sub_C444A0((__int64)(v18 + 24))) )
        return 0;
    }
  }
LABEL_6:
  if ( v7 == 33 )
  {
    if ( (unsigned __int8)sub_1178DE0(v60) )
    {
      v14 = *(_BYTE *)v5 == 42;
      v71 = v6;
      v72[0] = 0;
      if ( v14 && v6 == *(_QWORD *)(v5 - 64) && (unsigned __int8)sub_995B10(v72, *(_QWORD *)(v5 - 32)) )
      {
        HIDWORD(v68) = 0;
        v73 = 257;
        v15 = sub_AD64C0(*(_QWORD *)(v6 + 8), 1, 0);
        return sub_B33C40(a4, 0x173u, v6, v15, (unsigned int)v68, (__int64)&v71);
      }
    }
    return 0;
  }
  if ( !sub_B532A0(v7) )
    return 0;
  if ( v7 - 36 <= 1 )
  {
    sub_B52F50(v7);
    v31 = v6;
    v6 = v60;
    v60 = v31;
  }
  if ( *(_BYTE *)v5 != 44 || v60 != *(_QWORD *)(v5 - 64) || v6 != *(_QWORD *)(v5 - 32) )
  {
    v65 = 0;
    v64 = &v63;
    if ( !(unsigned __int8)sub_991580((__int64)&v64, v6) )
      goto LABEL_12;
    v32 = *((_DWORD *)v63 + 2);
    v67 = v32;
    if ( v32 > 0x40 )
    {
      sub_C43780((__int64)&v66, v63);
      v32 = v67;
      if ( v67 > 0x40 )
      {
        sub_C43D10((__int64)&v66);
        goto LABEL_59;
      }
      v33 = v66;
    }
    else
    {
      v33 = (unsigned __int64)*v63;
    }
    v34 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & ~v33;
    if ( !v32 )
      v34 = 0;
    v66 = v34;
LABEL_59:
    sub_C46250((__int64)&v66);
    v35 = v67;
    v67 = 0;
    v14 = *(_BYTE *)v5 == 42;
    v68 = v66;
    v69 = v35;
    v71 = v60;
    v72[0] = &v68;
    if ( !v14
      || v60 != *(_QWORD *)(v5 - 64)
      || (v36 = sub_10080A0((const void ***)v72, *(_QWORD *)(v5 - 32)), v35 = v69, !v36) )
    {
      v36 = 0;
    }
    if ( v35 > 0x40 && v68 )
    {
      v57 = v36;
      j_j___libc_free_0_0(v68);
      v36 = v57;
    }
    if ( v67 > 0x40 && v66 )
    {
      v58 = v36;
      j_j___libc_free_0_0(v66);
      v36 = v58;
    }
    if ( v36 )
      goto LABEL_47;
LABEL_12:
    if ( *(_BYTE *)v5 == 44 && v6 == *(_QWORD *)(v5 - 64) && v60 == *(_QWORD *)(v5 - 32) )
      goto LABEL_82;
    v64 = &v63;
    v65 = 0;
    v11 = sub_991580((__int64)&v64, v60);
    if ( !v11 )
      return 0;
    v37 = *((_DWORD *)v63 + 2);
    v67 = v37;
    if ( v37 > 0x40 )
    {
      sub_C43780((__int64)&v66, v63);
      v37 = v67;
      if ( v67 > 0x40 )
      {
        sub_C43D10((__int64)&v66);
        goto LABEL_74;
      }
      v38 = v66;
    }
    else
    {
      v38 = (unsigned __int64)*v63;
    }
    v39 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & ~v38;
    v14 = v37 == 0;
    v40 = 0;
    if ( !v14 )
      v40 = v39;
    v66 = v40;
LABEL_74:
    sub_C46250((__int64)&v66);
    v14 = *(_BYTE *)v5 == 42;
    v71 = v6;
    v41 = v67;
    v67 = 0;
    v68 = v66;
    v69 = v41;
    v72[0] = &v68;
    if ( v14 && v6 == *(_QWORD *)(v5 - 64) )
    {
      v14 = !sub_10080A0((const void ***)v72, *(_QWORD *)(v5 - 32));
      v41 = v69;
      if ( !v14 )
        v11 = 0;
    }
    if ( v41 > 0x40 && v68 )
      j_j___libc_free_0_0(v68);
    if ( v67 > 0x40 )
    {
      if ( v66 )
        j_j___libc_free_0_0(v66);
    }
    if ( v11 )
      return 0;
LABEL_82:
    v73 = 257;
    HIDWORD(v68) = 0;
    return sub_B33C40(a4, 0x173u, v6, v60, (unsigned int)v68, (__int64)&v71);
  }
LABEL_47:
  v22 = *(_QWORD *)(v5 + 16);
  if ( !v22 || *(_QWORD *)(v22 + 8) )
  {
    v23 = *(_QWORD *)(a1 + 16);
    if ( !v23 || *(_QWORD *)(v23 + 8) )
      return 0;
  }
  HIDWORD(v68) = 0;
  v73 = 257;
  v24 = sub_B33C40(a4, 0x173u, v6, v60, (unsigned int)v68, (__int64)&v71);
  v70 = 257;
  v25 = v24;
  v26 = sub_AD6530(*(_QWORD *)(v24 + 8), 371);
  result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a4 + 80) + 32LL))(
             *(_QWORD *)(a4 + 80),
             15,
             v26,
             v25,
             0,
             0);
  if ( !result )
  {
    v73 = 257;
    v61 = sub_B504D0(15, v26, v25, (__int64)&v71, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v61,
      &v68,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    v27 = *(_QWORD *)a4;
    result = v61;
    v28 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v28 )
    {
      do
      {
        v29 = *(_QWORD *)(v27 + 8);
        v30 = *(_DWORD *)v27;
        v27 += 16;
        v62 = result;
        sub_B99FD0(result, v30, v29);
        result = v62;
      }
      while ( v28 != v27 );
    }
  }
  return result;
}
