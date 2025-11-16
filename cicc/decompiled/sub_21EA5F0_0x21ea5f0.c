// Function: sub_21EA5F0
// Address: 0x21ea5f0
//
__int64 __fastcall sub_21EA5F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  _BYTE *v7; // r13
  size_t v8; // rax
  size_t v9; // r9
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __m128i si128; // xmm0
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  char v19; // cl
  unsigned __int64 v20; // r10
  int v21; // eax
  __int64 v22; // rax
  _QWORD *v23; // rdi
  unsigned int v24; // r10d
  int v25; // eax
  int v26; // [rsp+Ch] [rbp-B4h]
  char n; // [rsp+18h] [rbp-A8h]
  size_t na; // [rsp+18h] [rbp-A8h]
  __int64 v29; // [rsp+28h] [rbp-98h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v31; // [rsp+38h] [rbp-88h]
  _QWORD v32[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v33; // [rsp+50h] [rbp-70h]
  __int64 v34; // [rsp+58h] [rbp-68h]
  _QWORD v35[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v36; // [rsp+70h] [rbp-50h] BYREF
  __int64 v37; // [rsp+78h] [rbp-48h]
  _QWORD v38[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( **(_WORD **)(a3 + 16) != 1 )
  {
    *(_BYTE *)(a1 + 4) = 0;
    return a1;
  }
  v5 = *(_QWORD *)(a3 + 32);
  v30 = v32;
  v7 = *(_BYTE **)(v5 + 24);
  if ( !v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v8 = strlen(*(const char **)(v5 + 24));
  v36 = (_QWORD *)v8;
  v9 = v8;
  if ( v8 > 0xF )
  {
    na = v8;
    v22 = sub_22409D0(&v30, &v36, 0);
    v9 = na;
    v30 = (_QWORD *)v22;
    v23 = (_QWORD *)v22;
    v32[0] = v36;
LABEL_33:
    memcpy(v23, v7, v9);
    v8 = (size_t)v36;
    v10 = v30;
    goto LABEL_8;
  }
  if ( v8 != 1 )
  {
    if ( !v8 )
    {
      v10 = v32;
      goto LABEL_8;
    }
    v23 = v32;
    goto LABEL_33;
  }
  LOBYTE(v32[0]) = *v7;
  v10 = v32;
LABEL_8:
  v31 = v8;
  *((_BYTE *)v10 + v8) = 0;
  if ( *(_DWORD *)(a3 + 40) > 3u && (v11 = *(_QWORD *)(a3 + 32), *(_BYTE *)(v11 + 120) == 1) )
  {
    n = 1;
    v26 = *(_DWORD *)(v11 + 144);
  }
  else
  {
    n = 0;
  }
  v33 = v35;
  strcpy((char *)v35, "setmaxnreg.");
  v34 = 11;
  v12 = sub_22416F0(&v30, v35, 0, 11);
  if ( v12 != -1 )
  {
    v29 = 17;
    v36 = v38;
    v13 = v12 + v34 + 3;
    v14 = sub_22409D0(&v36, &v29, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_435F7B0);
    v36 = (_QWORD *)v14;
    v38[0] = v29;
    *(_BYTE *)(v14 + 16) = 50;
    *(__m128i *)v14 = si128;
    v37 = v29;
    *((_BYTE *)v36 + v29) = 0;
    if ( v13 == sub_22416F0(&v30, v36, v13, v37) )
    {
      v16 = v37 + v13;
      v17 = v16;
      if ( !n )
        goto LABEL_30;
      if ( v31 <= v16 )
      {
        if ( v31 <= v16 + 1 || *((_BYTE *)v30 + v16) != 36 || *((_BYTE *)v30 + v16 + 1) != 48 )
          goto LABEL_31;
        v18 = v16;
      }
      else
      {
        v18 = v16;
        while ( 1 )
        {
          v19 = *((_BYTE *)v30 + v18);
          v20 = v18++;
          if ( v19 != 9 && v19 != 32 )
            break;
          if ( v31 <= v18 )
            goto LABEL_19;
        }
        v18 = v20;
LABEL_19:
        if ( v31 <= v18 + 1 || *((_BYTE *)v30 + v18) != 36 || *((_BYTE *)v30 + v18 + 1) != 48 )
          goto LABEL_27;
      }
      if ( v31 <= v18 + 2 || (unsigned int)*((unsigned __int8 *)v30 + v18 + 2) - 48 > 9 )
      {
        *(_BYTE *)(a1 + 4) = 1;
        *(_DWORD *)a1 = v26;
        goto LABEL_25;
      }
LABEL_30:
      while ( v31 > v17 )
      {
LABEL_27:
        LOBYTE(v21) = *((_BYTE *)v30 + v17);
        if ( (_BYTE)v21 != 32 && (_BYTE)v21 != 9 )
        {
          v24 = 0;
          if ( (unsigned int)(unsigned __int8)v21 - 48 <= 9 )
          {
            while ( 1 )
            {
              v25 = (char)v21 - 48;
              if ( ~v25 / 0xAu < v24 )
                break;
              ++v17;
              v24 = v25 + 10 * v24;
              if ( v31 > v17 )
              {
                v21 = *((unsigned __int8 *)v30 + v17);
                if ( (unsigned int)(v21 - 48) <= 9 )
                  continue;
              }
              *(_BYTE *)(a1 + 4) = 1;
              *(_DWORD *)a1 = v24;
              goto LABEL_25;
            }
          }
          break;
        }
        ++v17;
      }
LABEL_31:
      *(_BYTE *)(a1 + 4) = 0;
LABEL_25:
      if ( v36 != v38 )
        j_j___libc_free_0(v36, v38[0] + 1LL);
      goto LABEL_37;
    }
    if ( v36 != v38 )
      j_j___libc_free_0(v36, v38[0] + 1LL);
  }
  *(_BYTE *)(a1 + 4) = 0;
LABEL_37:
  if ( v33 != v35 )
    j_j___libc_free_0(v33, v35[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  return a1;
}
