// Function: sub_121A7A0
// Address: 0x121a7a0
//
__int64 __fastcall sub_121A7A0(
        __int64 a1,
        unsigned __int64 a2,
        const void *a3,
        size_t a4,
        unsigned __int64 *a5,
        __int64 **a6)
{
  unsigned __int64 v10; // r9
  int v11; // eax
  __int64 v12; // rdi
  _BOOL4 v13; // r15d
  unsigned __int64 v15; // rax
  int v16; // eax
  char v17; // r14
  __int64 v18; // rsi
  int v19; // eax
  __int64 v20; // r9
  char v21; // al
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v25; // [rsp+18h] [rbp-E8h]
  __int64 *v26; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v27; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int64 v28; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD *v31; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v32; // [rsp+70h] [rbp-90h]
  char *v33; // [rsp+80h] [rbp-80h] BYREF
  __int64 v34; // [rsp+88h] [rbp-78h]
  _BYTE v35[16]; // [rsp+90h] [rbp-70h] BYREF
  char v36; // [rsp+A0h] [rbp-60h]
  char v37; // [rsp+A1h] [rbp-5Fh]

  v10 = *a5;
  if ( *a5 && !a5[1] )
  {
    v37 = 1;
    v33 = "redefinition of type";
    v13 = 1;
    v36 = 3;
    sub_11FD800(a1 + 176, a2, (__int64)&v33, 1);
    return v13;
  }
  v11 = *(_DWORD *)(a1 + 240);
  if ( v11 == 293 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    v15 = *a5;
    a5[1] = 0;
    if ( !v15 )
    {
      v15 = sub_BCC840(*(_QWORD **)a1, a3, a4);
      *a5 = v15;
    }
    *a6 = (__int64 *)v15;
    return 0;
  }
  if ( v11 != 10 )
  {
    if ( v11 != 8 )
    {
      v12 = a1 + 176;
      if ( !v10 )
      {
        *a6 = 0;
        v37 = 1;
        v33 = "expected type";
        v36 = 3;
        return (_BOOL4)sub_12190A0(a1, a6, (int *)&v33, 0);
      }
LABEL_7:
      v37 = 1;
      v33 = "forward references to non-struct type";
      v13 = 1;
      v36 = 3;
      sub_11FD800(v12, a2, (__int64)&v33, 1);
      return v13;
    }
    v17 = 0;
LABEL_16:
    a5[1] = 0;
    if ( !v10 )
    {
      v23 = sub_BCC840(*(_QWORD **)a1, a3, a4);
      *a5 = v23;
      v10 = v23;
    }
    v18 = (__int64)&v33;
    v25 = v10;
    v33 = v35;
    v34 = 0x800000000LL;
    v19 = sub_121A060(a1, (__int64)&v33);
    v20 = v25;
    v13 = v19;
    if ( (_BYTE)v19 || v17 && (v18 = 11, v21 = sub_120AFE0(a1, 11, "expected '>' in packed struct"), v20 = v25, v21) )
    {
      v13 = 1;
    }
    else
    {
      v18 = v20;
      v26 = (__int64 *)v20;
      sub_BD0A20(&v27, v20, (__int64 *)v33, (unsigned int)v34, v17 & 1);
      v22 = v27 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v27 = 0;
        v28 = v22 | 1;
        sub_C64870((__int64)v29, (__int64 *)&v28);
        v18 = *(_QWORD *)(a1 + 232);
        v32 = 260;
        v31 = v29;
        sub_11FD800(a1 + 176, v18, (__int64)&v31, 1);
        if ( (__int64 *)v29[0] != &v30 )
        {
          v18 = v30 + 1;
          j_j___libc_free_0(v29[0], v30 + 1);
        }
        if ( (v28 & 1) != 0 || (v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v28, v18);
        v13 = (v27 & 1) == 0;
        if ( (v27 & 1) != 0 || (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v27, v18);
      }
      else
      {
        *a6 = v26;
      }
    }
    if ( v33 != v35 )
      _libc_free(v33, v18);
    return v13;
  }
  v16 = sub_1205200(a1 + 176);
  v12 = a1 + 176;
  *(_DWORD *)(a1 + 240) = v16;
  if ( v16 == 8 )
  {
    v10 = *a5;
    v17 = 1;
    goto LABEL_16;
  }
  if ( *a5 )
    goto LABEL_7;
  *a6 = 0;
  return sub_121A490(a1, a6, 1u);
}
