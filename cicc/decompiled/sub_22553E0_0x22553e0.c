// Function: sub_22553E0
// Address: 0x22553e0
//
__int64 *__fastcall sub_22553E0(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5, _BYTE **a6)
{
  _BYTE *v6; // r13
  _BYTE *v9; // rdx
  __int128 *v11; // rax
  _DWORD *v12; // rax
  const char *v13; // rbp
  const char *v14; // r15
  char *v15; // rbp
  size_t v16; // rax
  size_t v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19[8]; // [rsp+8h] [rbp-40h] BYREF

  v6 = a1 + 2;
  v9 = a6[1];
  if ( a3 >= 0 && v9 )
  {
    v11 = sub_2254490();
    v12 = sub_22543D0((pthread_mutex_t *)v11, a3);
    if ( !v12 )
    {
      *a1 = (__int64)v6;
      sub_22552A0(a1, *a6, (__int64)&a6[1][(_QWORD)*a6]);
      return a1;
    }
    v13 = (const char *)*((_QWORD *)v12 + 1);
    v14 = *a6;
    __uselocale();
    v15 = dgettext(v13, v14);
    __uselocale();
    *a1 = (__int64)v6;
    if ( !v15 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v16 = strlen(v15);
    v19[0] = v16;
    v17 = v16;
    if ( v16 > 0xF )
    {
      v18 = sub_22409D0((__int64)a1, v19, 0);
      *a1 = v18;
      v6 = (_BYTE *)v18;
      a1[2] = v19[0];
    }
    else
    {
      if ( v16 == 1 )
      {
        *((_BYTE *)a1 + 16) = *v15;
LABEL_10:
        a1[1] = v16;
        v6[v16] = 0;
        return a1;
      }
      if ( !v16 )
        goto LABEL_10;
    }
    memcpy(v6, v15, v17);
    v16 = v19[0];
    v6 = (_BYTE *)*a1;
    goto LABEL_10;
  }
  *a1 = (__int64)v6;
  sub_22552A0(a1, *a6, (__int64)&v9[(_QWORD)*a6]);
  return a1;
}
