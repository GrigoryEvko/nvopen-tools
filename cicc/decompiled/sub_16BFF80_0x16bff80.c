// Function: sub_16BFF80
// Address: 0x16bff80
//
__int64 *__fastcall sub_16BFF80(__int64 *a1)
{
  _QWORD *v1; // r13
  int v2; // eax
  size_t v4; // rax
  unsigned int v5; // ebx
  __int64 v6; // rax
  size_t v7; // [rsp+8h] [rbp-1B8h] BYREF
  struct utsname name; // [rsp+10h] [rbp-1B0h] BYREF

  v1 = a1 + 2;
  v2 = uname(&name);
  *a1 = (__int64)(a1 + 2);
  if ( !v2 )
  {
    v4 = strlen(name.release);
    v7 = v4;
    v5 = v4;
    if ( v4 > 0xF )
    {
      v6 = sub_22409D0(a1, &v7, 0);
      *a1 = v6;
      v1 = (_QWORD *)v6;
      a1[2] = v7;
    }
    else
    {
      if ( v4 == 1 )
      {
        *((_BYTE *)a1 + 16) = name.release[0];
LABEL_7:
        a1[1] = v4;
        *((_BYTE *)v1 + v4) = 0;
        return a1;
      }
      if ( !v4 )
        goto LABEL_7;
    }
    if ( v5 >= 8 )
    {
      *v1 = *(_QWORD *)name.release;
      *(_QWORD *)((char *)v1 + v5 - 8) = *(_QWORD *)&name.nodename[v5 + 57];
      qmemcpy(
        (void *)((unsigned __int64)(v1 + 1) & 0xFFFFFFFFFFFFFFF8LL),
        (const void *)(name.release - ((char *)v1 - ((unsigned __int64)(v1 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
        8LL * ((v5 + (_DWORD)v1 - (((_DWORD)v1 + 8) & 0xFFFFFFF8)) >> 3));
    }
    else if ( (v5 & 4) != 0 )
    {
      *(_DWORD *)v1 = *(_DWORD *)name.release;
      *(_DWORD *)((char *)v1 + v5 - 4) = *(_DWORD *)&name.nodename[v5 + 61];
    }
    else if ( v5 )
    {
      *(_BYTE *)v1 = name.release[0];
      if ( (v5 & 2) != 0 )
        *(_WORD *)((char *)v1 + v5 - 2) = *(_WORD *)&name.nodename[v5 + 63];
    }
    v4 = v7;
    v1 = (_QWORD *)*a1;
    goto LABEL_7;
  }
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  return a1;
}
