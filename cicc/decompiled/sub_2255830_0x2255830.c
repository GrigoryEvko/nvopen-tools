// Function: sub_2255830
// Address: 0x2255830
//
volatile signed __int32 **__fastcall sub_2255830(
        volatile signed __int32 **a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        volatile signed __int32 **a6)
{
  __int128 *v9; // rax
  _DWORD *v10; // rax
  volatile signed __int32 *v11; // r14
  const char *v12; // rbp
  char *v13; // rbp
  size_t v14; // rax
  size_t v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r13
  volatile signed __int32 *v18; // rcx

  if ( a3 >= 0 && *((_QWORD *)*a6 - 3) && (v9 = sub_2254490(), (v10 = sub_22543D0((pthread_mutex_t *)v9, a3)) != 0) )
  {
    v11 = *a6;
    v12 = (const char *)*((_QWORD *)v10 + 1);
    __uselocale();
    v13 = dgettext(v12, (const char *)v11);
    __uselocale();
    if ( !v13 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v14 = strlen(v13);
    v15 = v14;
    if ( v14 )
    {
      v16 = sub_22153F0(v14, 0);
      v17 = v16;
      v18 = (volatile signed __int32 *)(v16 + 24);
      if ( v15 == 1 )
        *(_BYTE *)(v16 + 24) = *v13;
      else
        v18 = (volatile signed __int32 *)memcpy((void *)(v16 + 24), v13, v15);
      if ( (_UNKNOWN *)v17 != &unk_4FD67C0 )
      {
        *(_DWORD *)(v17 + 16) = 0;
        *(_QWORD *)v17 = v15;
        *(_BYTE *)(v17 + v15 + 24) = 0;
      }
    }
    else
    {
      v18 = (volatile signed __int32 *)&unk_4FD67D8;
    }
    *a1 = v18;
    return a1;
  }
  else
  {
    sub_2215E70(a1, a6);
    return a1;
  }
}
