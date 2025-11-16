// Function: sub_16F5140
// Address: 0x16f5140
//
void __fastcall sub_16F5140(__int64 a1, int a2, __int64 a3, double a4)
{
  __int64 v7; // r8
  char v8; // r13
  __int64 v9; // r8
  _QWORD *v10; // rdi
  __int64 v11; // rdi
  int v12; // r8d
  int v13; // r9d
  char *v14; // rax
  __int64 v15; // rax
  size_t v16; // rax
  _QWORD *v17; // rcx
  size_t v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  char *v23; // rcx
  char *v24; // r13
  unsigned int v25; // ecx
  unsigned int v26; // ecx
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // [rsp+10h] [rbp-B0h]
  double v30; // [rsp+18h] [rbp-A8h]
  char *format; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-98h]
  _BYTE v33[16]; // [rsp+30h] [rbp-90h] BYREF
  char s[32]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v35[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v36; // [rsp+70h] [rbp-50h]
  _WORD *v37; // [rsp+78h] [rbp-48h]
  int v38; // [rsp+80h] [rbp-40h]
  char **p_format; // [rsp+88h] [rbp-38h]

  v30 = a4;
  v7 = sub_16F5120(a2);
  if ( *(_BYTE *)(a3 + 8) )
    v7 = *(_QWORD *)a3;
  if ( fabs(a4) <= 1.797693134862316e308 )
  {
    v8 = 101;
    if ( a2 )
    {
      v8 = 69;
      if ( a2 != 1 )
        v8 = 102;
    }
    p_format = &format;
    v32 = 0x800000000LL;
    v29 = v7;
    v35[0] = &unk_49EFC48;
    format = v33;
    v38 = 1;
    v37 = 0;
    v36 = 0;
    v35[1] = 0;
    sub_16E7A40((__int64)v35, 0, 0, 0);
    v9 = v29;
    if ( (unsigned __int64)(v36 - (_QWORD)v37) <= 1 )
    {
      v20 = sub_16E7EE0((__int64)v35, "%.", 2u);
      v9 = v29;
      v10 = (_QWORD *)v20;
    }
    else
    {
      v10 = v35;
      *v37++ = 11813;
    }
    v11 = sub_16E7A90((__int64)v10, v9);
    v14 = *(char **)(v11 + 24);
    if ( (unsigned __int64)v14 >= *(_QWORD *)(v11 + 16) )
    {
      sub_16E7DE0(v11, v8);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v14 + 1;
      *v14 = v8;
    }
    if ( a2 == 3 )
      v30 = a4 * 100.0;
    v15 = (unsigned int)v32;
    if ( (unsigned int)v32 >= HIDWORD(v32) )
    {
      sub_16CD150((__int64)&format, v33, 0, 1, v12, v13);
      v15 = (unsigned int)v32;
    }
    format[v15] = 0;
    snprintf(s, 0x20u, format, v30, a4);
    v16 = strlen(s);
    v17 = *(_QWORD **)(a1 + 24);
    v18 = v16;
    if ( v16 > *(_QWORD *)(a1 + 16) - (_QWORD)v17 )
    {
      sub_16E7EE0(a1, s, v16);
      if ( a2 != 3 )
      {
LABEL_16:
        v35[0] = &unk_49EFD28;
        sub_16E7960((__int64)v35);
        if ( format != v33 )
          _libc_free((unsigned __int64)format);
        return;
      }
    }
    else
    {
      if ( v16 )
      {
        if ( (unsigned int)v16 >= 8 )
        {
          v22 = (unsigned __int64)(v17 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v17 = *(_QWORD *)s;
          *(_QWORD *)((char *)v17 + (unsigned int)v16 - 8) = *(_QWORD *)&v33[(unsigned int)v16 + 8];
          v23 = (char *)v17 - v22;
          v24 = (char *)(s - v23);
          v25 = (v16 + (_DWORD)v23) & 0xFFFFFFF8;
          if ( v25 >= 8 )
          {
            v26 = v25 & 0xFFFFFFF8;
            v27 = 0;
            do
            {
              v28 = v27;
              v27 += 8;
              *(_QWORD *)(v22 + v28) = *(_QWORD *)&v24[v28];
            }
            while ( v27 < v26 );
          }
        }
        else if ( (v16 & 4) != 0 )
        {
          *(_DWORD *)v17 = *(_DWORD *)s;
          *(_DWORD *)((char *)v17 + (unsigned int)v16 - 4) = *(_DWORD *)&v33[(unsigned int)v16 + 12];
        }
        else if ( (_DWORD)v16 )
        {
          *(_BYTE *)v17 = s[0];
          if ( (v16 & 2) != 0 )
            *(_WORD *)((char *)v17 + (unsigned int)v16 - 2) = *(_WORD *)&v33[(unsigned int)v16 + 14];
        }
        *(_QWORD *)(a1 + 24) += v18;
      }
      if ( a2 != 3 )
        goto LABEL_16;
    }
    v19 = *(_BYTE **)(a1 + 24);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(a1 + 16) )
    {
      sub_16E7DE0(a1, 37);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v19 + 1;
      *v19 = 37;
    }
    goto LABEL_16;
  }
  v21 = *(_QWORD *)(a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v21) <= 2 )
  {
    sub_16E7EE0(a1, "INF", 3u);
  }
  else
  {
    *(_BYTE *)(v21 + 2) = 70;
    *(_WORD *)v21 = 20041;
    *(_QWORD *)(a1 + 24) += 3LL;
  }
}
