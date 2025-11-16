// Function: sub_C83B00
// Address: 0xc83b00
//
void __fastcall sub_C83B00(_QWORD *a1, struct passwd *p_resultbuf)
{
  __int64 v2; // r14
  _BYTE *v3; // rbx
  unsigned __int64 v4; // r14
  char *v6; // rbx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // r15
  size_t v10; // rbx
  __int64 v11; // rcx
  char *v12; // rax
  size_t v13; // rcx
  char *v14; // r14
  size_t v15; // rdi
  struct passwd *v16; // rax
  const char *pw_dir; // r13
  size_t v18; // rax
  __int64 v19; // rdi
  size_t v20; // rbx
  _BYTE *v21; // rbx
  _BYTE *v22; // rax
  __int64 v23; // rcx
  struct passwd *v24; // r10
  struct passwd *v25; // r11
  struct passwd *v26; // r14
  size_t v27; // r13
  struct passwd *v28; // r9
  size_t v29; // rdx
  size_t v30; // rdx
  size_t v31; // r13
  struct passwd *v32; // rdi
  size_t v33; // r15
  __int64 v34; // [rsp+0h] [rbp-220h]
  size_t buflen; // [rsp+8h] [rbp-218h]
  size_t buflena; // [rsp+8h] [rbp-218h]
  size_t n; // [rsp+10h] [rbp-210h]
  size_t na; // [rsp+10h] [rbp-210h]
  size_t nb; // [rsp+10h] [rbp-210h]
  _BYTE *v40; // [rsp+18h] [rbp-208h]
  struct passwd *v41; // [rsp+18h] [rbp-208h]
  struct passwd *v42; // [rsp+18h] [rbp-208h]
  struct passwd *v43; // [rsp+18h] [rbp-208h]
  struct passwd *v44; // [rsp+18h] [rbp-208h]
  struct passwd *v45; // [rsp+18h] [rbp-208h]
  unsigned __int64 v46; // [rsp+20h] [rbp-200h]
  size_t v47; // [rsp+20h] [rbp-200h]
  struct passwd *v48; // [rsp+20h] [rbp-200h]
  struct passwd *v49; // [rsp+20h] [rbp-200h]
  struct passwd *v50; // [rsp+20h] [rbp-200h]
  struct passwd *v51; // [rsp+20h] [rbp-200h]
  _BYTE *v52; // [rsp+20h] [rbp-200h]
  struct passwd *result; // [rsp+38h] [rbp-1E8h] BYREF
  char *name; // [rsp+40h] [rbp-1E0h] BYREF
  _QWORD v55[2]; // [rsp+50h] [rbp-1D0h] BYREF
  _QWORD v56[4]; // [rsp+60h] [rbp-1C0h] BYREF
  __int16 v57; // [rsp+80h] [rbp-1A0h]
  char v58[32]; // [rsp+90h] [rbp-190h] BYREF
  __int16 v59; // [rsp+B0h] [rbp-170h]
  char v60[32]; // [rsp+C0h] [rbp-160h] BYREF
  __int16 v61; // [rsp+E0h] [rbp-140h]
  char v62[32]; // [rsp+F0h] [rbp-130h] BYREF
  __int16 v63; // [rsp+110h] [rbp-110h]
  struct passwd resultbuf; // [rsp+120h] [rbp-100h] BYREF
  _BYTE *v65; // [rsp+150h] [rbp-D0h] BYREF
  size_t v66; // [rsp+158h] [rbp-C8h]
  unsigned __int64 v67; // [rsp+160h] [rbp-C0h]
  _BYTE v68[184]; // [rsp+168h] [rbp-B8h] BYREF

  v2 = a1[1];
  v3 = (_BYTE *)*a1;
  if ( v2 && *v3 == 126 )
  {
    v4 = v2 - 1;
    v40 = v3 + 1;
    v6 = v3 + 1;
    v7 = v4;
    if ( v4 )
    {
      while ( 1 )
      {
        p_resultbuf = (struct passwd *)(unsigned int)*v6;
        if ( sub_C80E10((__int64)&v65, *v6) )
          break;
        ++v6;
        if ( !--v7 )
          goto LABEL_46;
      }
      v8 = v4 - v7;
      if ( v4 - v7 > v4 )
        v8 = v4;
      v9 = v8;
      v46 = v8 + 1;
      if ( v8 + 1 > v4 )
      {
        v46 = v4;
        v10 = 0;
      }
      else
      {
        v10 = v4 - v46;
      }
    }
    else
    {
LABEL_46:
      v46 = v4;
      v9 = v4;
      v10 = 0;
    }
    v66 = 0;
    v65 = v68;
    v67 = 128;
    if ( v9 )
    {
      v11 = sysconf(70);
      if ( v11 <= 0 )
        v11 = 0x4000;
      n = v11;
      v12 = (char *)sub_2207820(v11);
      v13 = n;
      v14 = v12;
      if ( v12 )
      {
        memset(v12, 0, n);
        v13 = n;
      }
      buflen = v13;
      name = (char *)v55;
      sub_C7FBF0((__int64 *)&name, v40, (__int64)&v40[v9]);
      p_resultbuf = &resultbuf;
      result = 0;
      getpwnam_r(name, &resultbuf, v14, buflen, &result);
      if ( result && result->pw_dir )
      {
        v66 = 0;
        v15 = 0;
        if ( v10 > v67 )
        {
          sub_C8D290(&v65, v68, v10, 1);
          v15 = v66;
        }
        if ( v10 )
        {
          memcpy(&v65[v15], &v40[v46], v10);
          v15 = v66;
        }
        v16 = result;
        a1[1] = 0;
        v66 = v10 + v15;
        pw_dir = v16->pw_dir;
        v18 = strlen(pw_dir);
        v19 = 0;
        v20 = v18;
        if ( v18 > a1[2] )
        {
          sub_C8D290(a1, a1 + 3, v18, 1);
          v19 = a1[1];
        }
        if ( v20 )
        {
          memcpy((void *)(*a1 + v19), pw_dir, v20);
          v19 = a1[1];
        }
        a1[1] = v19 + v20;
        v63 = 257;
        v61 = 257;
        v56[0] = v65;
        v59 = 257;
        v57 = 261;
        p_resultbuf = (struct passwd *)v56;
        v56[1] = v66;
        sub_C81B70(a1, (__int64)v56, (__int64)v58, (__int64)v60, (__int64)v62);
      }
      if ( name != (char *)v55 )
      {
        p_resultbuf = (struct passwd *)(v55[0] + 1LL);
        j_j___libc_free_0(name, v55[0] + 1LL);
      }
      if ( v14 )
        j_j___libc_free_0_0(v14);
    }
    else if ( (unsigned __int8)sub_C83840(&v65) )
    {
      *(_BYTE *)*a1 = *v65;
      v21 = v65;
      v22 = (_BYTE *)*a1;
      v23 = a1[1];
      v24 = (struct passwd *)(v65 + 1);
      v25 = (struct passwd *)&v65[v66];
      v26 = (struct passwd *)(*a1 + 1LL);
      v27 = v66 - 1;
      v28 = (struct passwd *)(*a1 + v23);
      v29 = v23 + v66 - 1;
      if ( v26 == v28 )
      {
        if ( v29 > a1[2] )
        {
          p_resultbuf = (struct passwd *)(a1 + 3);
          v43 = (struct passwd *)(v65 + 1);
          v49 = (struct passwd *)&v65[v66];
          sub_C8D290(a1, a1 + 3, v29, 1);
          v23 = a1[1];
          v24 = v43;
          v25 = v49;
          v28 = (struct passwd *)(v23 + *a1);
        }
        if ( v24 != v25 )
        {
          p_resultbuf = v24;
          memcpy(v28, v24, v27);
          v23 = a1[1];
        }
        a1[1] = v27 + v23;
      }
      else
      {
        if ( v29 > a1[2] )
        {
          p_resultbuf = (struct passwd *)(a1 + 3);
          v42 = (struct passwd *)(v65 + 1);
          v48 = (struct passwd *)&v65[v66];
          sub_C8D290(a1, a1 + 3, v29, 1);
          v22 = (_BYTE *)*a1;
          v23 = a1[1];
          v24 = v42;
          v25 = v48;
          v26 = (struct passwd *)(*a1 + 1LL);
          v28 = (struct passwd *)(*a1 + v23);
        }
        v30 = v23 - 1;
        if ( v23 - 1 >= v27 )
        {
          v32 = v28;
          v33 = v23 - v27;
          if ( v27 + v23 > a1[2] )
          {
            p_resultbuf = (struct passwd *)(a1 + 3);
            nb = (size_t)v24;
            v45 = v28;
            v52 = v22;
            sub_C8D290(a1, a1 + 3, v27 + v23, 1);
            v23 = a1[1];
            v24 = (struct passwd *)nb;
            v28 = v45;
            v22 = v52;
            v32 = (struct passwd *)(v23 + *a1);
          }
          if ( v27 )
          {
            p_resultbuf = (struct passwd *)&v22[v33];
            v44 = v24;
            v51 = v28;
            memmove(v32, &v22[v33], v27);
            v23 = a1[1];
            v24 = v44;
            v28 = v51;
          }
          a1[1] = v27 + v23;
          if ( v33 != 1 )
          {
            p_resultbuf = v26;
            v50 = v24;
            memmove((char *)&v28->pw_name - v33 + 1, v26, v33 - 1);
            v24 = v50;
          }
          if ( v27 )
          {
            p_resultbuf = v24;
            memmove(v26, v24, v27);
          }
        }
        else
        {
          v31 = v23 + v27;
          a1[1] = v31;
          if ( v28 != v26 )
          {
            p_resultbuf = v26;
            v34 = v23;
            buflena = (size_t)v24;
            na = (size_t)v28;
            v41 = v25;
            v47 = v23 - 1;
            memcpy(&v22[v31 - v30], v26, v30);
            v23 = v34;
            v24 = (struct passwd *)buflena;
            v28 = (struct passwd *)na;
            v25 = v41;
            v30 = v47;
          }
          if ( v30 )
          {
            do
            {
              *((_BYTE *)&v26->pw_name + v9) = v21[v9 + 1];
              ++v9;
            }
            while ( v30 != v9 );
            v24 = (struct passwd *)&v21[v23];
          }
          if ( v25 != v24 )
          {
            p_resultbuf = v24;
            memcpy(v28, v24, (char *)v25 - (char *)v24);
          }
        }
      }
    }
    if ( v65 != v68 )
      _libc_free(v65, p_resultbuf);
  }
}
