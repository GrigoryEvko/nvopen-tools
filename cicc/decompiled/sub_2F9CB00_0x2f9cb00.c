// Function: sub_2F9CB00
// Address: 0x2f9cb00
//
void __fastcall sub_2F9CB00(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, char a5)
{
  unsigned __int8 *v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 *v12; // rdx
  unsigned __int8 **v13; // rax
  __int64 v14; // rcx
  unsigned __int8 *v15; // rsi
  unsigned __int8 **v16; // rdx
  unsigned __int8 **v17; // rax
  __int64 *j; // rdx
  char v19; // dl
  __int64 v20; // rax
  char v21; // al
  unsigned __int8 *i; // rdx
  __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rax
  unsigned __int8 *v27; // [rsp+0h] [rbp-E0h]
  unsigned __int8 *v28; // [rsp+0h] [rbp-E0h]
  unsigned __int8 v29; // [rsp+8h] [rbp-D8h]
  unsigned __int8 *v30; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+8h] [rbp-D8h]
  __int64 *v32; // [rsp+8h] [rbp-D8h]
  __int64 v33; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int8 *v34; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v36; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 **v37; // [rsp+38h] [rbp-A8h]
  __int64 v38; // [rsp+40h] [rbp-A0h]
  int v39; // [rsp+48h] [rbp-98h]
  char v40; // [rsp+4Ch] [rbp-94h]
  char v41; // [rsp+50h] [rbp-90h] BYREF
  __int64 v42[2]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v43; // [rsp+70h] [rbp-70h]
  unsigned __int64 v44; // [rsp+78h] [rbp-68h]
  unsigned __int8 **v45; // [rsp+80h] [rbp-60h]
  unsigned __int64 *v46; // [rsp+88h] [rbp-58h]
  __int64 *v47; // [rsp+90h] [rbp-50h]
  __int64 v48; // [rsp+98h] [rbp-48h]
  __int64 v49; // [rsp+A0h] [rbp-40h]
  __int64 v50; // [rsp+A8h] [rbp-38h]

  v33 = a2;
  v37 = (unsigned __int8 **)&v41;
  v36 = 0;
  v38 = 2;
  v39 = 0;
  v40 = 1;
  v42[0] = 0;
  v42[1] = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  sub_2785050(v42, 0);
  v11 = v47;
  if ( v47 == (__int64 *)(v49 - 8) )
  {
    sub_2785520((unsigned __int64 *)v42, &v33);
    v12 = v47;
  }
  else
  {
    if ( v47 )
    {
      *v47 = v33;
      v11 = v47;
    }
    v12 = v11 + 1;
    v47 = v11 + 1;
  }
  v13 = (unsigned __int8 **)v43;
  v14 = (__int64)&v35;
  if ( v12 != (__int64 *)v43 )
  {
    do
    {
      v15 = *v13;
      v16 = v45 - 1;
      v34 = *v13;
      if ( v13 == v45 - 1 )
      {
        j_j___libc_free_0(v44);
        v15 = v34;
        v16 = (unsigned __int8 **)(*++v46 + 512);
        v44 = *v46;
        v45 = v16;
        v43 = v44;
      }
      else
      {
        v43 = (unsigned __int64)(v13 + 1);
      }
      if ( !v40 )
        goto LABEL_17;
      v17 = v37;
      v14 = HIDWORD(v38);
      v16 = &v37[HIDWORD(v38)];
      if ( v37 != v16 )
      {
        while ( *v17 != v15 )
        {
          if ( v16 == ++v17 )
            goto LABEL_32;
        }
        goto LABEL_13;
      }
LABEL_32:
      if ( HIDWORD(v38) < (unsigned int)v38 )
      {
        v14 = (unsigned int)++HIDWORD(v38);
        *v16 = v15;
        ++v36;
      }
      else
      {
LABEL_17:
        sub_C8CC70((__int64)&v36, (__int64)v15, (__int64)v16, v14, (__int64)v9, v10);
        if ( !v19 )
          goto LABEL_13;
      }
      v20 = *((_QWORD *)v34 + 2);
      if ( !v20 || *(_QWORD *)(v20 + 8) )
        goto LABEL_13;
      if ( a5 )
      {
        v29 = *v34;
        if ( (unsigned int)*v34 - 30 <= 0xA )
          goto LABEL_13;
        v27 = v34;
        v21 = sub_B46970(v34);
        v14 = v29 & 0xFD;
        if ( (v29 & 0xFD) == 0x54 || v21 )
          goto LABEL_13;
        if ( (unsigned __int8)sub_B46420((__int64)v27) )
        {
          if ( *((_QWORD *)v27 + 5) == *(_QWORD *)(a4 + 40) )
          {
            for ( i = v27 + 24; ; i = (unsigned __int8 *)*((_QWORD *)v30 + 1) )
            {
              if ( i )
              {
                v23 = (__int64)(i - 24);
                if ( (unsigned __int8 *)a4 == i - 24 )
                  goto LABEL_35;
              }
              else
              {
                v23 = 0;
              }
              v30 = i;
              if ( (unsigned __int8)sub_B46490(v23) )
                break;
            }
          }
          goto LABEL_13;
        }
      }
LABEL_35:
      v31 = sub_FDD860(*(__int64 **)(a1 + 40), *(_QWORD *)(v33 + 40));
      if ( v31 <= sub_FDD860(*(__int64 **)(a1 + 40), *((_QWORD *)v34 + 5)) )
      {
        v14 = a3[8];
        v24 = (_QWORD *)a3[6];
        if ( v24 == (_QWORD *)(v14 - 8) )
        {
          sub_2785520(a3, &v34);
          v9 = v34;
        }
        else
        {
          v9 = v34;
          if ( v24 )
          {
            *v24 = v34;
            v24 = (_QWORD *)a3[6];
            v9 = v34;
          }
          a3[6] = (unsigned __int64)(v24 + 1);
        }
        v25 = 4LL * (*((_DWORD *)v9 + 1) & 0x7FFFFFF);
        if ( (v9[7] & 0x40) != 0 )
        {
          v26 = (__int64 *)*((_QWORD *)v9 - 1);
          v9 = (unsigned __int8 *)&v26[v25];
        }
        else
        {
          v26 = (__int64 *)&v9[-(v25 * 8)];
        }
        for ( j = v47; v9 != (unsigned __int8 *)v26; v26 += 4 )
        {
          v14 = *v26;
          if ( *(_BYTE *)*v26 > 0x1Cu )
          {
            v35 = *v26;
            if ( j == (__int64 *)(v49 - 8) )
            {
              v28 = v9;
              v32 = v26;
              sub_2785520((unsigned __int64 *)v42, &v35);
              j = v47;
              v9 = v28;
              v26 = v32;
            }
            else
            {
              if ( j )
              {
                *j = v14;
                j = v47;
              }
              v47 = ++j;
            }
          }
        }
        goto LABEL_14;
      }
LABEL_13:
      j = v47;
LABEL_14:
      v13 = (unsigned __int8 **)v43;
    }
    while ( (__int64 *)v43 != j );
  }
  sub_2784FD0((unsigned __int64 *)v42);
  if ( !v40 )
    _libc_free((unsigned __int64)v37);
}
