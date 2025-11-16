// Function: sub_11FCC80
// Address: 0x11fcc80
//
__int64 *__fastcall sub_11FCC80(__int64 *a1, double a2)
{
  _QWORD *v2; // rbx
  unsigned int v3; // r13d
  const char *v4; // r14
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  double v9; // xmm0_8
  size_t v10; // rax
  __int64 v11; // rax
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 2;
  if ( a2 <= 1.0 )
  {
    if ( a2 < 0.0 )
    {
      LOBYTE(v3) = 7;
      *a1 = (__int64)v2;
      v4 = "#3d50c3";
      v12[0] = 7;
      v5 = 7;
      goto LABEL_4;
    }
    v9 = round(a2 * 99.0);
    *a1 = (__int64)v2;
    v4 = &a3d50c3[8 * (int)v9];
    v10 = strlen(v4);
    v12[0] = v10;
    v3 = v10;
    if ( v10 <= 0xF )
    {
      if ( v10 == 1 )
      {
        *((_BYTE *)a1 + 16) = *v4;
        goto LABEL_8;
      }
      if ( !v10 )
        goto LABEL_8;
    }
    else
    {
      v11 = sub_22409D0(a1, v12, 0);
      *a1 = v11;
      v2 = (_QWORD *)v11;
      a1[2] = v12[0];
    }
  }
  else
  {
    *a1 = (__int64)v2;
    v3 = 7;
    v4 = "#b70d28";
    v12[0] = 7;
  }
  v5 = v3;
  if ( v3 >= 8 )
  {
    *v2 = *(_QWORD *)v4;
    *(_QWORD *)((char *)v2 + v3 - 8) = *(_QWORD *)&v4[v3 - 8];
    qmemcpy(
      (void *)((unsigned __int64)(v2 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v4 - ((const char *)v2 - ((unsigned __int64)(v2 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * ((v3 + (_DWORD)v2 - (((_DWORD)v2 + 8) & 0xFFFFFFF8)) >> 3));
    goto LABEL_8;
  }
LABEL_4:
  if ( (v3 & 4) != 0 )
  {
    *(_DWORD *)v2 = *(_DWORD *)v4;
    *(_DWORD *)((char *)v2 + v5 - 4) = *(_DWORD *)&v4[v5 - 4];
  }
  else if ( (_DWORD)v5 )
  {
    *(_BYTE *)v2 = *v4;
    if ( (v5 & 2) != 0 )
      *(_WORD *)((char *)v2 + v5 - 2) = *(_WORD *)&v4[v5 - 2];
  }
LABEL_8:
  v6 = v12[0];
  v7 = *a1;
  a1[1] = v12[0];
  *(_BYTE *)(v7 + v6) = 0;
  return a1;
}
