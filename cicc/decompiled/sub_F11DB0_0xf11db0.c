// Function: sub_F11DB0
// Address: 0xf11db0
//
__int64 __fastcall sub_F11DB0(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v2; // r15
  __int64 result; // rax
  __int64 v4; // r13
  unsigned __int8 *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // r14
  unsigned __int8 *v10; // r9
  __int64 *v11; // rcx
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r12
  _BYTE *v21; // rax
  unsigned __int64 v22; // rax
  __int64 i; // r12
  char *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 v29; // r12
  __int64 v30; // rsi
  __int64 v31; // rbx
  int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // r14
  __int64 v39; // r13
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 *v43; // r14
  __int64 v44; // rax
  unsigned __int8 *v45; // rax
  __int64 v46; // r15
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // [rsp+8h] [rbp-D8h]
  __int64 v50; // [rsp+8h] [rbp-D8h]
  __int64 *v52; // [rsp+20h] [rbp-C0h]
  __int64 v53; // [rsp+20h] [rbp-C0h]
  __int64 v54; // [rsp+20h] [rbp-C0h]
  __int64 *v55; // [rsp+28h] [rbp-B8h]
  __int64 v56; // [rsp+28h] [rbp-B8h]
  __int64 v57; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v58; // [rsp+30h] [rbp-B0h]
  __int64 v59; // [rsp+30h] [rbp-B0h]
  __int64 v60; // [rsp+38h] [rbp-A8h]
  __int64 v61; // [rsp+38h] [rbp-A8h]
  __int64 v62; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v63; // [rsp+48h] [rbp-98h] BYREF
  const char *v64; // [rsp+50h] [rbp-90h] BYREF
  __int16 v65; // [rsp+70h] [rbp-70h]
  __int64 *v66; // [rsp+80h] [rbp-60h] BYREF
  __int64 v67; // [rsp+88h] [rbp-58h]
  _BYTE v68[16]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-40h]

  v2 = *((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v2 != 84 )
    return 0;
  v4 = *((_QWORD *)a2 - 4);
  v5 = a2;
  if ( *(_BYTE *)v4 != 84 )
    return 0;
  v6 = *(_QWORD *)(v2 + 16);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 8) )
    return 0;
  v7 = *(_QWORD *)(v4 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFF) != (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) )
    return 0;
  v8 = *((_QWORD *)a2 + 5);
  if ( v8 != *(_QWORD *)(v2 + 40) || *(_QWORD *)(v4 + 40) != v8 )
    return 0;
  v9 = a1;
  v10 = sub_AD93D0((unsigned int)*a2 - 29, *((_QWORD *)a2 + 1), 0, 0);
  if ( !v10 )
    goto LABEL_23;
  v66 = (__int64 *)v68;
  v67 = 0x400000000LL;
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
  {
    v11 = *(__int64 **)(v4 - 8);
    v12 = *(_DWORD *)(v4 + 4);
  }
  else
  {
    v12 = *(_DWORD *)(v4 + 4);
    v11 = (__int64 *)(v4 - 32LL * (v12 & 0x7FFFFFF));
  }
  v13 = v12 & 0x7FFFFFF;
  v14 = 4LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v15 = (__int64 *)(v2 - v14 * 8);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v15 = *(__int64 **)(v2 - 8);
  v55 = &v15[v14];
  if ( &v15[v14] != v15 )
  {
    v58 = v10;
    v52 = &v11[4 * v13];
    v16 = v11;
    do
    {
      if ( v16 == v52 )
        break;
      v17 = 32LL * *(unsigned int *)(v4 + 72);
      if ( *(_QWORD *)(*(_QWORD *)(v2 - 8)
                     + 32LL * *(unsigned int *)(v2 + 72)
                     + 8LL * (unsigned int)(((__int64)v15 - *(_QWORD *)(v2 - 8)) >> 5)) != *(_QWORD *)(*(_QWORD *)(v4 - 8) + v17 + 8LL * (unsigned int)(((__int64)v16 - *(_QWORD *)(v4 - 8)) >> 5)) )
        goto LABEL_21;
      v25 = *v15;
      v26 = *v16;
      if ( v58 == (unsigned __int8 *)*v15 )
      {
        v48 = (unsigned int)v67;
        v10 = (unsigned __int8 *)((unsigned int)v67 + 1LL);
        if ( (unsigned __int64)v10 > HIDWORD(v67) )
        {
          v50 = *v16;
          sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 8u, v26, (__int64)v10);
          v48 = (unsigned int)v67;
          v26 = v50;
        }
        v66[v48] = v26;
        LODWORD(v67) = v67 + 1;
      }
      else
      {
        if ( v58 != (unsigned __int8 *)v26 )
        {
LABEL_21:
          v5 = a2;
          v9 = a1;
          if ( v66 != (__int64 *)v68 )
            _libc_free(v66, v17);
LABEL_23:
          if ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFF) != 2 || (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 2 )
            return 0;
          v66 = &v62;
          if ( (unsigned __int8)sub_F11D70(&v66, **(_BYTE ***)(v2 - 8)) )
          {
            v18 = *(_QWORD *)(v2 - 8);
            v19 = 32LL * *(unsigned int *)(v2 + 72);
            v20 = *(_QWORD *)(v18 + v19 + 8);
            v59 = *(_QWORD *)(v18 + v19);
          }
          else
          {
            v66 = &v62;
            if ( !(unsigned __int8)sub_F11D70(&v66, *(_BYTE **)(*(_QWORD *)(v2 - 8) + 32LL)) )
              return 0;
            v40 = *(_QWORD *)(v2 - 8);
            v41 = 32LL * *(unsigned int *)(v2 + 72);
            v20 = *(_QWORD *)(v40 + v41);
            v59 = *(_QWORD *)(v40 + v41 + 8);
          }
          v66 = (__int64 *)&v63;
          v21 = (_BYTE *)sub_F0A930(v4, v59);
          if ( !(unsigned __int8)sub_F11D70(&v66, v21) )
            return 0;
          v22 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v22 == v20 + 48 )
            goto LABEL_72;
          if ( !v22 )
            BUG();
          v53 = v22 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 > 0xA )
LABEL_72:
            BUG();
          if ( *(_BYTE *)(v22 - 24) != 31
            || (*(_DWORD *)(v22 - 20) & 0x7FFFFFF) == 3
            || !(unsigned __int8)sub_B192B0(v9[10], v20) )
          {
            return 0;
          }
          v56 = v20;
          for ( i = *(_QWORD *)(*((_QWORD *)v5 + 5) + 56LL); ; i = *(_QWORD *)(i + 8) )
          {
            if ( i )
            {
              v24 = (char *)(i - 24);
              if ( v5 == (unsigned __int8 *)(i - 24) )
              {
                v42 = v56;
                v57 = sub_96E6C0((unsigned int)*v5 - 29, v62, v63, v9[11]);
                if ( v57 )
                {
                  sub_D5F1F0(v9[4], v53);
                  v43 = (__int64 *)v9[4];
                  v69 = 257;
                  v54 = sub_F0A930(v4, v42);
                  v44 = sub_F0A930(v2, v42);
                  v45 = (unsigned __int8 *)sub_F0A990(
                                             v43,
                                             (unsigned int)*v5 - 29,
                                             v44,
                                             v54,
                                             (int)v64,
                                             0,
                                             (__int64)&v66,
                                             0);
                  v46 = (__int64)v45;
                  if ( (unsigned __int8)(*v45 - 42) <= 0x11u )
                    sub_B45260(v45, (__int64)v5, 1);
                  v47 = *((_QWORD *)v5 + 1);
                  v69 = 257;
                  v61 = sub_F0A7C0(v47, 2, (const char **)&v66, 0, 0);
                  sub_F0A850(v61, v46, v42);
                  sub_F0A850(v61, v57, v59);
                  return v61;
                }
                return 0;
              }
            }
            else
            {
              v24 = 0;
            }
            if ( !(unsigned __int8)sub_98CD80(v24) )
              return 0;
          }
        }
        v27 = (unsigned int)v67;
        v28 = (unsigned int)v67 + 1LL;
        if ( v28 > HIDWORD(v67) )
        {
          v49 = *v15;
          sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 8u, v28, (__int64)v10);
          v27 = (unsigned int)v67;
          v25 = v49;
        }
        v66[v27] = v25;
        LODWORD(v67) = v67 + 1;
      }
      v15 += 4;
      v16 += 4;
    }
    while ( v55 != v15 );
  }
  v65 = 257;
  v29 = 0;
  v30 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
  result = sub_F0A7C0(*(_QWORD *)(v2 + 8), v30, &v64, 0, 0);
  v31 = result;
  if ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFF) != 0 )
  {
    do
    {
      v37 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
      v38 = *(_QWORD *)(*(_QWORD *)(v2 - 8) + 32LL * *(unsigned int *)(v2 + 72) + 8 * v29);
      v39 = v66[v29];
      if ( v37 == *(_DWORD *)(v31 + 72) )
      {
        sub_B48D90(v31);
        v37 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
      }
      v32 = (v37 + 1) & 0x7FFFFFF;
      v33 = v32 | *(_DWORD *)(v31 + 4) & 0xF8000000;
      v34 = *(_QWORD *)(v31 - 8) + 32LL * (unsigned int)(v32 - 1);
      *(_DWORD *)(v31 + 4) = v33;
      if ( *(_QWORD *)v34 )
      {
        v35 = *(_QWORD *)(v34 + 8);
        **(_QWORD **)(v34 + 16) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 16) = *(_QWORD *)(v34 + 16);
      }
      *(_QWORD *)v34 = v39;
      if ( v39 )
      {
        v36 = *(_QWORD *)(v39 + 16);
        *(_QWORD *)(v34 + 8) = v36;
        if ( v36 )
        {
          v30 = v34 + 8;
          *(_QWORD *)(v36 + 16) = v34 + 8;
        }
        *(_QWORD *)(v34 + 16) = v39 + 16;
        *(_QWORD *)(v39 + 16) = v34;
      }
      ++v29;
      *(_QWORD *)(*(_QWORD *)(v31 - 8)
                + 32LL * *(unsigned int *)(v31 + 72)
                + 8LL * ((*(_DWORD *)(v31 + 4) & 0x7FFFFFFu) - 1)) = v38;
    }
    while ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFFu) > (unsigned int)v29 );
    result = v31;
  }
  if ( v66 != (__int64 *)v68 )
  {
    v60 = result;
    _libc_free(v66, v30);
    return v60;
  }
  return result;
}
