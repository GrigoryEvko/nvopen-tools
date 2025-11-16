// Function: sub_B16E20
// Address: 0xb16e20
//
__int64 __fastcall sub_B16E20(__int64 a1, _BYTE *a2, __int64 a3, _QWORD *a4)
{
  unsigned __int8 *v6; // rbx
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  _BYTE **v9; // rax
  _BYTE *v10; // rax
  unsigned __int8 v11; // dl
  unsigned __int8 v12; // dl
  const char **v13; // rax
  const char *v14; // rdi
  __int64 v15; // rdx
  __int64 result; // rax
  unsigned __int8 *v17; // rdi
  __int64 v18; // rdx
  size_t v19; // rcx
  __int64 v20; // rsi
  size_t v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-1B8h]
  __int64 v23; // [rsp+18h] [rbp-1A8h]
  unsigned __int8 *v24; // [rsp+20h] [rbp-1A0h] BYREF
  size_t n; // [rsp+28h] [rbp-198h]
  unsigned __int8 src[16]; // [rsp+30h] [rbp-190h] BYREF
  _QWORD v27[2]; // [rsp+40h] [rbp-180h] BYREF
  __int128 v28; // [rsp+50h] [rbp-170h]
  char v29; // [rsp+60h] [rbp-160h]
  char v30; // [rsp+61h] [rbp-15Fh]
  _QWORD v31[2]; // [rsp+70h] [rbp-150h] BYREF
  __int128 v32; // [rsp+80h] [rbp-140h]
  char v33; // [rsp+90h] [rbp-130h]
  char v34; // [rsp+91h] [rbp-12Fh]
  _OWORD v35[2]; // [rsp+A0h] [rbp-120h] BYREF
  char v36; // [rsp+C0h] [rbp-100h]
  char v37; // [rsp+C1h] [rbp-FFh]
  _QWORD v38[4]; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v39; // [rsp+F0h] [rbp-D0h]
  __int128 v40; // [rsp+100h] [rbp-C0h]
  __int16 v41; // [rsp+120h] [rbp-A0h]
  __int128 v42; // [rsp+130h] [rbp-90h]
  char v43; // [rsp+150h] [rbp-70h]
  char v44; // [rsp+151h] [rbp-6Fh]
  __int128 v45; // [rsp+160h] [rbp-60h]
  __int64 v46; // [rsp+180h] [rbp-40h]

  v6 = (unsigned __int8 *)(a1 + 48);
  *(_QWORD *)a1 = a1 + 16;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  sub_B157E0(a1 + 64, a4);
  if ( !*a4 )
    return sub_2241130(a1 + 32, 0, *(_QWORD *)(a1 + 40), "<UNKNOWN LOCATION>", 18);
  v44 = 1;
  LOWORD(v46) = 265;
  LODWORD(v45) = sub_B10CF0((__int64)a4);
  *(_QWORD *)&v42 = ":";
  v43 = 3;
  v41 = 265;
  LODWORD(v40) = sub_B10CE0((__int64)a4);
  v7 = sub_B10CD0((__int64)a4);
  v8 = *(_BYTE *)(v7 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(_BYTE ***)(v7 - 32);
  else
    v9 = (_BYTE **)(v7 - 16 - 8LL * ((v8 >> 2) & 0xF));
  v10 = *v9;
  if ( *v10 == 16 )
    goto LABEL_7;
  v11 = *(v10 - 16);
  if ( (v11 & 2) == 0 )
  {
    v10 = *(_BYTE **)&v10[-8 * ((v11 >> 2) & 0xF) - 16];
    if ( v10 )
      goto LABEL_7;
LABEL_21:
    v15 = 0;
    v14 = byte_3F871B3;
    goto LABEL_11;
  }
  v10 = (_BYTE *)**((_QWORD **)v10 - 4);
  if ( !v10 )
    goto LABEL_21;
LABEL_7:
  v12 = *(v10 - 16);
  if ( (v12 & 2) != 0 )
    v13 = (const char **)*((_QWORD *)v10 - 4);
  else
    v13 = (const char **)&v10[-8 * ((v12 >> 2) & 0xF) - 16];
  v14 = *v13;
  if ( *v13 )
    v14 = (const char *)sub_B91420(v14, a4);
  else
    v15 = 0;
LABEL_11:
  v38[0] = v14;
  LOWORD(v39) = 773;
  v38[1] = v15;
  v38[2] = ":";
  v37 = v41;
  *(_QWORD *)&v35[0] = v38;
  v35[1] = v40;
  v36 = 2;
  v31[0] = v35;
  v34 = v43;
  v31[1] = v23;
  v32 = v42;
  v33 = 2;
  v27[0] = v31;
  v28 = v45;
  v27[1] = v22;
  v29 = 2;
  v30 = v46;
  sub_CA0F50(&v24, v27);
  result = (__int64)v24;
  v17 = *(unsigned __int8 **)(a1 + 32);
  if ( v24 == src )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        result = src[0];
        *v17 = src[0];
      }
      else
      {
        result = (__int64)memcpy(v17, src, n);
      }
      v21 = n;
      v17 = *(unsigned __int8 **)(a1 + 32);
    }
    *(_QWORD *)(a1 + 40) = v21;
    v17[v21] = 0;
    v17 = v24;
    goto LABEL_15;
  }
  v18 = *(_QWORD *)src;
  v19 = n;
  if ( v6 == v17 )
  {
    *(_QWORD *)(a1 + 32) = v24;
    *(_QWORD *)(a1 + 40) = v19;
    *(_QWORD *)(a1 + 48) = v18;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 32) = v24;
    *(_QWORD *)(a1 + 40) = v19;
    *(_QWORD *)(a1 + 48) = v18;
    if ( v17 )
    {
      v24 = v17;
      *(_QWORD *)src = v20;
      goto LABEL_15;
    }
  }
  v24 = src;
  v17 = src;
LABEL_15:
  n = 0;
  *v17 = 0;
  if ( v24 != src )
    return j_j___libc_free_0(v24, *(_QWORD *)src + 1LL);
  return result;
}
