// Function: sub_1BBD340
// Address: 0x1bbd340
//
__int64 __fastcall sub_1BBD340(__int64 ***a1)
{
  __int64 **v1; // r12
  __int64 **v2; // rsi
  unsigned int v3; // eax
  __int64 ***v4; // rax
  __int64 ***v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // r15
  __int64 *v11; // r12
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 *v14; // rax
  unsigned __int64 v15; // r12
  _QWORD *v16; // r15
  _QWORD *v17; // r14
  __int64 *v19; // r9
  __int64 *v20; // rdi
  __int64 v21; // rcx
  __int64 ***v22; // rdx
  __int64 ***v23; // rdi
  __int64 v24; // rax
  __int64 ***v25; // rax
  __int64 ***v26; // r13
  __int64 **v27; // rdx
  __int64 ***v28; // r12
  int v29; // r8d
  _QWORD *v30; // r9
  __int64 v31; // rax
  __int64 ***v32; // rax
  _QWORD *v33; // [rsp+8h] [rbp-F8h]
  unsigned int v34; // [rsp+18h] [rbp-E8h]
  unsigned int v35; // [rsp+1Ch] [rbp-E4h]
  __int64 **v37; // [rsp+28h] [rbp-D8h]
  __int64 **v38; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v39; // [rsp+30h] [rbp-D0h]
  __int64 v40; // [rsp+38h] [rbp-C8h]
  __int64 v41; // [rsp+40h] [rbp-C0h]
  _BYTE *v42; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+58h] [rbp-A8h]
  _BYTE v44[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v46; // [rsp+88h] [rbp-78h]
  __int64 ***v47; // [rsp+90h] [rbp-70h]
  __int64 v48; // [rsp+98h] [rbp-68h]
  int v49; // [rsp+A0h] [rbp-60h]
  _BYTE v50[88]; // [rsp+A8h] [rbp-58h] BYREF

  v1 = *a1;
  v2 = a1[1];
  v45 = 0;
  v3 = *((_DWORD *)v1 + 2);
  v48 = 4;
  v34 = v3;
  v4 = (__int64 ***)v50;
  v46 = (__int64 *)v50;
  v47 = (__int64 ***)v50;
  v49 = 0;
  if ( v2 == v1 )
    return 0;
  v5 = (__int64 ***)v50;
  v6 = 0;
  v35 = 0;
  do
  {
    while ( 1 )
    {
      v7 = **v1;
      if ( *(_BYTE *)(v7 + 16) <= 0x17u )
        goto LABEL_4;
      if ( v6 )
        break;
      v6 = **v1;
LABEL_4:
      v1 += 22;
      if ( v2 == v1 )
        goto LABEL_29;
    }
    if ( v4 == v5 )
    {
      v21 = HIDWORD(v48);
      v22 = &v4[v21];
      if ( &v4[v21] == v4 )
      {
LABEL_71:
        v4 = &v5[v21];
        v23 = &v5[v21];
      }
      else
      {
        while ( *v4 != (__int64 **)v6 )
        {
          if ( v22 == ++v4 )
            goto LABEL_71;
        }
        v23 = &v5[v21];
      }
    }
    else
    {
      v4 = (__int64 ***)sub_16CC9F0((__int64)&v45, v6);
      if ( *v4 == (__int64 **)v6 )
      {
        if ( v47 == (__int64 ***)v46 )
          v23 = &v47[HIDWORD(v48)];
        else
          v23 = &v47[(unsigned int)v48];
      }
      else
      {
        if ( v47 != (__int64 ***)v46 )
          goto LABEL_10;
        v4 = &v47[HIDWORD(v48)];
        v23 = v4;
      }
    }
    if ( v23 != v4 )
    {
      *v4 = (__int64 **)-2LL;
      ++v49;
    }
LABEL_10:
    v8 = 3LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
    {
      v9 = *(__int64 **)(v6 - 8);
      v10 = (__int64)&v9[v8];
    }
    else
    {
      v10 = v6;
      v9 = (__int64 *)(v6 - v8 * 8);
    }
    if ( v9 != (__int64 *)v10 )
    {
      v38 = v1;
      v11 = (__int64 *)v10;
      v12 = v9;
      v40 = v7;
      while ( 1 )
      {
        v13 = *v12;
        if ( *(_BYTE *)(*v12 + 16) <= 0x17u || !sub_1BBCD20((__int64)a1, *v12) )
          goto LABEL_14;
        v14 = v46;
        if ( v47 == (__int64 ***)v46 )
        {
          v19 = &v46[HIDWORD(v48)];
          if ( v46 == v19 )
          {
LABEL_74:
            if ( HIDWORD(v48) >= (unsigned int)v48 )
              goto LABEL_18;
            ++HIDWORD(v48);
            *v19 = v13;
            ++v45;
          }
          else
          {
            v20 = 0;
            while ( v13 != *v14 )
            {
              if ( *v14 == -2 )
                v20 = v14;
              if ( v19 == ++v14 )
              {
                if ( !v20 )
                  goto LABEL_74;
                *v20 = v13;
                --v49;
                ++v45;
                break;
              }
            }
          }
LABEL_14:
          v12 += 3;
          if ( v11 == v12 )
            goto LABEL_19;
        }
        else
        {
LABEL_18:
          v12 += 3;
          sub_16CCBA0((__int64)&v45, v13);
          if ( v11 == v12 )
          {
LABEL_19:
            v7 = v40;
            v1 = v38;
            break;
          }
        }
      }
    }
    if ( v6 + 24 != (*(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v37 = v1;
      v15 = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v16 = (_QWORD *)v6;
      v17 = (_QWORD *)(v6 + 24);
      v41 = v7;
      do
      {
        while ( v17 == (_QWORD *)(v16[5] + 40LL) )
        {
          v17 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v41 + 40) + 40LL) & 0xFFFFFFFFFFFFFFF8LL);
          if ( (_QWORD *)v15 == v17 )
            goto LABEL_27;
        }
        if ( !v17 )
          BUG();
        if ( *((_BYTE *)v17 - 8) == 78 )
        {
          v24 = *(v17 - 6);
          if ( (*(_BYTE *)(v24 + 16)
             || (*(_BYTE *)(v24 + 33) & 0x20) == 0
             || (unsigned int)(*(_DWORD *)(v24 + 36) - 35) > 3)
            && v16 != v17 - 3 )
          {
            v42 = v44;
            v43 = 0x400000000LL;
            v25 = v47;
            if ( v47 == (__int64 ***)v46 )
              v26 = &v47[HIDWORD(v48)];
            else
              v26 = &v47[(unsigned int)v48];
            if ( v47 != v26 )
            {
              while ( 1 )
              {
                v27 = *v25;
                if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v26 == ++v25 )
                  goto LABEL_59;
              }
              if ( v26 != v25 )
              {
                v39 = v15;
                v28 = v25;
                do
                {
                  v30 = sub_16463B0(*v27, v34);
                  v31 = (unsigned int)v43;
                  if ( (unsigned int)v43 >= HIDWORD(v43) )
                  {
                    v33 = v30;
                    sub_16CD150((__int64)&v42, v44, 0, 8, v29, (int)v30);
                    v31 = (unsigned int)v43;
                    v30 = v33;
                  }
                  *(_QWORD *)&v42[8 * v31] = v30;
                  v32 = v28 + 1;
                  LODWORD(v43) = v43 + 1;
                  if ( v28 + 1 == v26 )
                    break;
                  while ( 1 )
                  {
                    v27 = *v32;
                    v28 = v32;
                    if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v26 == ++v32 )
                      goto LABEL_68;
                  }
                }
                while ( v26 != v32 );
LABEL_68:
                v15 = v39;
              }
            }
LABEL_59:
            v35 += sub_14A36B0((__int64)a1[165]);
            if ( v42 != v44 )
              _libc_free((unsigned __int64)v42);
          }
        }
        v17 = (_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
      }
      while ( (_QWORD *)v15 != v17 );
LABEL_27:
      v7 = v41;
      v1 = v37;
    }
    v5 = v47;
    v4 = (__int64 ***)v46;
    v6 = v7;
    v1 += 22;
  }
  while ( v2 != v1 );
LABEL_29:
  if ( v5 != v4 )
    _libc_free((unsigned __int64)v5);
  return v35;
}
