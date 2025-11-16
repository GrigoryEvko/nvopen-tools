// Function: sub_16C33F0
// Address: 0x16c33f0
//
__int64 __fastcall sub_16C33F0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  ssize_t v6; // r14
  const char *v8; // rax
  size_t v9; // rax
  __int64 v10; // rdx
  size_t v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdi
  char s[8]; // [rsp+0h] [rbp-10C0h] BYREF
  __int64 v15; // [rsp+8h] [rbp-10B8h]
  char v16[128]; // [rsp+10h] [rbp-10B0h] BYREF
  char buf[4144]; // [rsp+90h] [rbp-1030h] BYREF

  sub_2241E40(a1, a2, a3, a4, a5);
  if ( !a3 )
    return 0;
  *(_DWORD *)(a3 + 8) = 0;
  if ( !byte_4FA04F8 && (unsigned int)sub_2207590(&byte_4FA04F8) )
  {
    byte_4FA0500 = access("/proc/self/fd", 4) == 0;
    sub_2207640(&byte_4FA04F8);
  }
  if ( byte_4FA0500 )
  {
    snprintf(s, 0x40u, "/proc/self/fd/%d", *a2);
    v6 = readlink(s, buf, 0x1000u);
    if ( v6 > 0 )
    {
      v13 = *(unsigned int *)(a3 + 8);
      if ( v6 > (unsigned __int64)*(unsigned int *)(a3 + 12) - v13 )
      {
        sub_16CD150(a3, a3 + 16, v6 + v13, 1);
        v13 = *(unsigned int *)(a3 + 8);
      }
      memcpy((void *)(*(_QWORD *)a3 + v13), buf, v6);
      *(_DWORD *)(a3 + 8) += v6;
    }
    return 0;
  }
  v15 = 0x8000000000LL;
  *(_QWORD *)s = v16;
  v8 = (const char *)sub_16E32E0(a1, s);
  if ( realpath(v8, buf) )
  {
    v9 = strlen(buf);
    v10 = *(unsigned int *)(a3 + 8);
    v11 = v9;
    if ( v9 > (unsigned __int64)*(unsigned int *)(a3 + 12) - v10 )
    {
      sub_16CD150(a3, a3 + 16, v9 + v10, 1);
      v10 = *(unsigned int *)(a3 + 8);
    }
    if ( !v11 )
      goto LABEL_15;
    v12 = (_QWORD *)(v10 + *(_QWORD *)a3);
    if ( (unsigned int)v11 >= 8 )
    {
      *v12 = *(_QWORD *)buf;
      *(_QWORD *)((char *)v12 + (unsigned int)v11 - 8) = *(_QWORD *)&v16[(unsigned int)v11 + 120];
      qmemcpy(
        (void *)((unsigned __int64)(v12 + 1) & 0xFFFFFFFFFFFFFFF8LL),
        (const void *)(buf - ((char *)v12 - ((unsigned __int64)(v12 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
        8LL * (((unsigned int)v11 + (_DWORD)v12 - (((_DWORD)v12 + 8) & 0xFFFFFFF8)) >> 3));
    }
    else
    {
      if ( (v11 & 4) != 0 )
      {
        *(_DWORD *)v12 = *(_DWORD *)buf;
        *(_DWORD *)((char *)v12 + (unsigned int)v11 - 4) = *(_DWORD *)&v16[(unsigned int)v11 + 124];
        LODWORD(v10) = *(_DWORD *)(a3 + 8);
        goto LABEL_15;
      }
      if ( !(_DWORD)v11 )
      {
LABEL_15:
        *(_DWORD *)(a3 + 8) = v10 + v11;
        goto LABEL_16;
      }
      *(_BYTE *)v12 = buf[0];
      if ( (v11 & 2) != 0 )
      {
        *(_WORD *)((char *)v12 + (unsigned int)v11 - 2) = *(_WORD *)&v16[(unsigned int)v11 + 126];
        LODWORD(v10) = *(_DWORD *)(a3 + 8);
        goto LABEL_15;
      }
    }
    LODWORD(v10) = *(_DWORD *)(a3 + 8);
    goto LABEL_15;
  }
LABEL_16:
  if ( *(char **)s != v16 )
    _libc_free(*(unsigned __int64 *)s);
  return 0;
}
