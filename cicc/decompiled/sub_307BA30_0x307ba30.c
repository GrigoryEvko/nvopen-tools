// Function: sub_307BA30
// Address: 0x307ba30
//
char *__fastcall sub_307BA30(__int64 a1, __int64 a2)
{
  __int16 v2; // ax
  __int64 v3; // rax
  __int64 v5; // rax
  _BYTE *v6; // r14
  char *v7; // rax
  size_t v8; // r15
  _QWORD *v9; // rdx
  __int64 v10; // rax
  char v11; // r14
  unsigned int v12; // ebx
  __int64 v13; // rax
  char *v14; // rax
  __m128i si128; // xmm0
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  char v18; // cl
  size_t v19; // r9
  int v20; // eax
  _QWORD *v21; // rdi
  unsigned int v22; // r9d
  int v23; // eax
  unsigned __int64 v24; // [rsp+8h] [rbp-B8h]
  size_t v25; // [rsp+18h] [rbp-A8h]
  size_t v26; // [rsp+28h] [rbp-98h] BYREF
  _QWORD *v27; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v28; // [rsp+38h] [rbp-88h]
  _QWORD v29[2]; // [rsp+40h] [rbp-80h] BYREF
  char *v30; // [rsp+50h] [rbp-70h]
  __int64 v31; // [rsp+58h] [rbp-68h]
  char v32[16]; // [rsp+60h] [rbp-60h] BYREF
  char *v33; // [rsp+70h] [rbp-50h] BYREF
  size_t v34; // [rsp+78h] [rbp-48h]
  _QWORD v35[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_WORD *)(a2 + 68);
  if ( v2 != 1 )
  {
    if ( (unsigned __int16)(v2 - 2622) > 1u )
    {
      BYTE4(v33) = 0;
    }
    else
    {
      v3 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
      BYTE4(v33) = 1;
      LODWORD(v33) = v3;
    }
    return v33;
  }
  v5 = *(_QWORD *)(a2 + 32);
  v27 = v29;
  v6 = *(_BYTE **)(v5 + 24);
  if ( !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v7 = (char *)strlen(*(const char **)(v5 + 24));
  v33 = v7;
  v8 = (size_t)v7;
  if ( (unsigned __int64)v7 > 0xF )
  {
    v27 = (_QWORD *)sub_22409D0((__int64)&v27, (unsigned __int64 *)&v33, 0);
    v21 = v27;
    v29[0] = v33;
    goto LABEL_43;
  }
  if ( v7 != (char *)1 )
  {
    if ( !v7 )
    {
      v9 = v29;
      goto LABEL_9;
    }
    v21 = v29;
LABEL_43:
    memcpy(v21, v6, v8);
    v7 = v33;
    v9 = v27;
    goto LABEL_9;
  }
  LOBYTE(v29[0]) = *v6;
  v9 = v29;
LABEL_9:
  v28 = (unsigned __int64)v7;
  v7[(_QWORD)v9] = 0;
  *(_WORD *)((char *)&v25 + 5) = 0;
  HIBYTE(v25) = 0;
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFFu) > 3 && (v10 = *(_QWORD *)(a2 + 32), v11 = *(_BYTE *)(v10 + 120), v11 == 1) )
  {
    v12 = *(_DWORD *)(v10 + 144);
  }
  else
  {
    v11 = 0;
    v12 = 0;
  }
  LODWORD(v25) = v12;
  BYTE4(v25) = v11;
  v30 = v32;
  strcpy(v32, "setmaxnreg.");
  v31 = 11;
  v13 = sub_22416F0((__int64 *)&v27, v32, 0, 0xBu);
  if ( v13 != -1 )
  {
    v26 = 17;
    v33 = (char *)v35;
    v24 = v13 + v31 + 3;
    v14 = (char *)sub_22409D0((__int64)&v33, &v26, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_435F7B0);
    v33 = v14;
    v35[0] = v26;
    v14[16] = 50;
    *(__m128i *)v14 = si128;
    v34 = v26;
    v33[v26] = 0;
    if ( v24 == sub_22416F0((__int64 *)&v27, v33, v24, v34) )
    {
      v16 = v24 + v34;
      if ( v11 )
      {
        if ( v16 >= v28 )
        {
          if ( v16 + 1 >= v28 || *((_BYTE *)v27 + v16) != 36 || *((_BYTE *)v27 + v16 + 1) != 48 )
          {
LABEL_29:
            v11 = 0;
            goto LABEL_30;
          }
          v17 = v24 + v34;
        }
        else
        {
          v17 = v24 + v34;
          while ( 1 )
          {
            v18 = *((_BYTE *)v27 + v17);
            v19 = v17++;
            if ( v18 != 9 && v18 != 32 )
              break;
            if ( v28 <= v17 )
              goto LABEL_20;
          }
          v17 = v19;
LABEL_20:
          if ( v28 <= v17 + 1 || *((_BYTE *)v27 + v17) != 36 || *((_BYTE *)v27 + v17 + 1) != 48 )
          {
LABEL_26:
            while ( 1 )
            {
              LOBYTE(v20) = *((_BYTE *)v27 + v16);
              if ( (_BYTE)v20 != 9 && (_BYTE)v20 != 32 )
                break;
              if ( ++v16 >= v28 )
                goto LABEL_29;
            }
            v11 = 0;
            v22 = 0;
            if ( (unsigned int)(unsigned __int8)v20 - 48 > 9 )
              goto LABEL_30;
            while ( 1 )
            {
              v23 = (char)v20 - 48;
              if ( v22 > ~v23 / 0xAu )
                goto LABEL_29;
              ++v16;
              v22 = v23 + 10 * v22;
              if ( v28 > v16 )
              {
                v20 = *((unsigned __int8 *)v27 + v16);
                if ( (unsigned int)(v20 - 48) <= 9 )
                  continue;
              }
              v12 = v22;
              v11 = 1;
              goto LABEL_30;
            }
          }
        }
        if ( v17 + 2 >= v28 || (unsigned int)*((unsigned __int8 *)v27 + v17 + 2) - 48 > 9 )
        {
          v26 = v25;
LABEL_30:
          if ( v33 != (char *)v35 )
            j_j___libc_free_0((unsigned __int64)v33);
          goto LABEL_38;
        }
      }
      if ( v16 < v28 )
        goto LABEL_26;
      goto LABEL_29;
    }
    if ( v33 != (char *)v35 )
      j_j___libc_free_0((unsigned __int64)v33);
  }
  v11 = 0;
LABEL_38:
  if ( v30 != v32 )
    j_j___libc_free_0((unsigned __int64)v30);
  LODWORD(v26) = v12;
  BYTE4(v26) = v11;
  v33 = (char *)v26;
  if ( v27 != v29 )
    j_j___libc_free_0((unsigned __int64)v27);
  return v33;
}
