// Function: sub_C7FF50
// Address: 0xc7ff50
//
__int64 __fastcall sub_C7FF50(__int64 a1, _DWORD *a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  ssize_t v6; // rax
  size_t v7; // r14
  const char *v9; // rax
  char *v10; // rsi
  size_t v11; // rax
  __int64 v12; // rdx
  size_t v13; // r15
  char *v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rdi
  signed __int64 v17; // rax
  char *v18; // rsi
  char s[8]; // [rsp+0h] [rbp-10D0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-10C8h]
  __int64 v21; // [rsp+10h] [rbp-10C0h]
  char v22[136]; // [rsp+18h] [rbp-10B8h] BYREF
  char buf[4144]; // [rsp+A0h] [rbp-1030h] BYREF

  sub_2241E40(a1, a2, a3, a4, a5);
  if ( !a3 )
    return 0;
  a3[1] = 0;
  if ( !byte_4F84108 && (unsigned int)sub_2207590(&byte_4F84108) )
  {
    byte_4F84110 = access("/proc/self/fd", 4) == 0;
    sub_2207640(&byte_4F84108);
  }
  if ( byte_4F84110 )
  {
    snprintf(s, 0x40u, "/proc/self/fd/%d", *a2);
    v6 = readlink(s, buf, 0x1000u);
    v7 = v6;
    if ( v6 > 0 )
    {
      v15 = a3[1];
      if ( (unsigned __int64)(v6 + v15) > a3[2] )
      {
        sub_C8D290(a3, a3 + 3, v6 + v15, 1);
        v15 = a3[1];
      }
      memcpy((void *)(*a3 + v15), buf, v7);
      a3[1] += v7;
    }
    return 0;
  }
  v20 = 0;
  *(_QWORD *)s = v22;
  v21 = 128;
  v9 = (const char *)sub_CA12A0(a1, s);
  v10 = buf;
  if ( realpath(v9, buf) )
  {
    v11 = strlen(buf);
    v12 = a3[1];
    v13 = v11;
    if ( v11 + v12 > a3[2] )
    {
      v10 = (char *)(a3 + 3);
      sub_C8D290(a3, a3 + 3, v11 + v12, 1);
      v12 = a3[1];
    }
    if ( !v13 )
      goto LABEL_15;
    v14 = (char *)(v12 + *a3);
    if ( (unsigned int)v13 >= 8 )
    {
      v16 = (_BYTE *)((unsigned __int64)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      *(_QWORD *)v14 = *(_QWORD *)buf;
      *(_QWORD *)&v14[(unsigned int)v13 - 8] = *(_QWORD *)&v22[(unsigned int)v13 + 128];
      v17 = v14 - v16;
      v18 = &buf[-v17];
      LODWORD(v17) = (unsigned int)(v13 + v17) >> 3;
      qmemcpy(v16, v18, 8LL * (unsigned int)v17);
      v10 = &v18[8 * (unsigned int)v17];
    }
    else
    {
      if ( (v13 & 4) != 0 )
      {
        *(_DWORD *)v14 = *(_DWORD *)buf;
        *(_DWORD *)&v14[(unsigned int)v13 - 4] = *(_DWORD *)&v22[(unsigned int)v13 + 132];
        v12 = a3[1];
        goto LABEL_15;
      }
      if ( !(_DWORD)v13 )
      {
LABEL_15:
        a3[1] = v12 + v13;
        goto LABEL_16;
      }
      *v14 = buf[0];
      if ( (v13 & 2) != 0 )
      {
        *(_WORD *)&v14[(unsigned int)v13 - 2] = *(_WORD *)&v22[(unsigned int)v13 + 134];
        v12 = a3[1];
        goto LABEL_15;
      }
    }
    v12 = a3[1];
    goto LABEL_15;
  }
LABEL_16:
  if ( *(char **)s != v22 )
    _libc_free(*(_QWORD *)s, v10);
  return 0;
}
