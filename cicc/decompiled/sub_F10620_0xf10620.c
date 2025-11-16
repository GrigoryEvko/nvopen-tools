// Function: sub_F10620
// Address: 0xf10620
//
__int64 __fastcall sub_F10620(const __m128i *a1, unsigned __int8 *a2)
{
  unsigned __int8 v3; // cl
  unsigned __int8 *v4; // r8
  char *v5; // rax
  unsigned __int8 v6; // dl
  __int64 v8; // r13
  unsigned int v9; // r14d
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // r11
  __int64 v18; // rdi
  unsigned __int8 *v19; // r15
  __int64 v20; // rdx
  char v21; // al
  __int64 *v22; // r11
  __int64 v23; // rdx
  int v24; // r12d
  __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // esi
  _BYTE *v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-B0h]
  int v31; // [rsp+8h] [rbp-A8h]
  __int64 *v32; // [rsp+8h] [rbp-A8h]
  __int64 *v33; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-A0h]
  char v35; // [rsp+1Ch] [rbp-94h]
  char *v36; // [rsp+20h] [rbp-90h] BYREF
  char v37; // [rsp+40h] [rbp-70h]
  char v38; // [rsp+41h] [rbp-6Fh]
  _BYTE v39[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v40; // [rsp+70h] [rbp-40h]

  v3 = *a2;
  v4 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v5 = (char *)*((_QWORD *)a2 - 4);
  if ( *a2 != 44 )
  {
    v5 = (char *)*((_QWORD *)a2 - 8);
    v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  }
  v6 = *v5;
  if ( (unsigned __int8)*v5 <= 0x1Cu )
    return 0;
  if ( v6 == 69 )
  {
    v8 = *((_QWORD *)v5 - 4);
    if ( !v8 )
      return 0;
    v35 = 1;
    v9 = 40;
  }
  else
  {
    if ( v6 != 68 )
      return 0;
    v8 = *((_QWORD *)v5 - 4);
    if ( !v8 )
      return 0;
    v35 = 0;
    v9 = 39;
  }
  v10 = *v4;
  v11 = *((_QWORD *)v5 + 2);
  if ( (unsigned __int8)v10 <= 0x1Cu )
  {
    if ( v11 )
    {
      if ( !*(_QWORD *)(v11 + 8) && (unsigned __int8)v10 <= 0x15u )
      {
        v34 = v4;
        v12 = sub_AD4C30((unsigned __int64)v4, *(__int64 ***)(v8 + 8), 0);
        v13 = sub_96F480(v9, v12, *((_QWORD *)v34 + 1), a1[5].m128i_i64[1]);
        if ( v34 == (unsigned __int8 *)v13 && v13 != 0 )
        {
          if ( v12 )
          {
            v3 = *a2;
            goto LABEL_26;
          }
        }
      }
    }
    return 0;
  }
  if ( (_BYTE)v10 != 68 && (_BYTE)v10 != 69 )
    return 0;
  v12 = *((_QWORD *)v4 - 4);
  if ( !v12 || *(_QWORD *)(v8 + 8) != *(_QWORD *)(v12 + 8) || v9 != v10 - 29 )
    return 0;
  if ( !v11 || *(_QWORD *)(v11 + 8) )
  {
    v14 = *((_QWORD *)v4 + 2);
    if ( !v14 || *(_QWORD *)(v14 + 8) )
      return 0;
  }
LABEL_26:
  if ( v3 == 44 )
  {
    v15 = v8;
    v8 = v12;
    v12 = v15;
  }
  if ( !sub_F0C210(a1, v3 - 29, v8, v12, (__int64)a2, v35 & 1) )
    return 0;
  v16 = *a2;
  if ( LOBYTE(qword_4F8BA48[8]) )
  {
    if ( (_BYTE)v16 == 42 )
    {
      v29 = (_BYTE *)*((_QWORD *)a2 - 8);
      if ( *v29 == 69 && *((_QWORD *)v29 - 4) && **((_BYTE **)a2 - 4) == 17 )
        return 0;
    }
  }
  v17 = a1[2].m128i_i64[0];
  v38 = 1;
  v37 = 3;
  v18 = *(_QWORD *)(v17 + 80);
  v36 = "narrow";
  v30 = v17;
  v31 = v16 - 29;
  v19 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v18 + 16LL))(
                             v18,
                             (unsigned int)(v16 - 29),
                             v8,
                             v12);
  if ( !v19 )
  {
    v40 = 257;
    v19 = (unsigned __int8 *)sub_B504D0(v31, v8, v12, (__int64)v39, 0, 0);
    v21 = sub_920620((__int64)v19);
    v22 = (__int64 *)v30;
    if ( v21 )
    {
      v23 = *(_QWORD *)(v30 + 96);
      v24 = *(_DWORD *)(v30 + 104);
      if ( v23 )
      {
        sub_B99FD0((__int64)v19, 3u, v23);
        v22 = (__int64 *)v30;
      }
      v32 = v22;
      sub_B45150((__int64)v19, v24);
      v22 = v32;
    }
    v33 = v22;
    (*(void (__fastcall **)(__int64, unsigned __int8 *, char **, __int64, __int64))(*(_QWORD *)v22[11] + 16LL))(
      v22[11],
      v19,
      &v36,
      v22[7],
      v22[8]);
    v25 = *v33;
    v26 = *v33 + 16LL * *((unsigned int *)v33 + 2);
    if ( *v33 != v26 )
    {
      do
      {
        v27 = *(_QWORD *)(v25 + 8);
        v28 = *(_DWORD *)v25;
        v25 += 16;
        sub_B99FD0((__int64)v19, v28, v27);
      }
      while ( v26 != v25 );
    }
  }
  if ( (unsigned __int8)(*v19 - 42) <= 0x11u )
  {
    if ( v35 )
      sub_B44850(v19, 1);
    else
      sub_B447F0(v19, 1);
  }
  v20 = *((_QWORD *)a2 + 1);
  v40 = 257;
  return sub_B51D30(v9, (__int64)v19, v20, (__int64)v39, 0, 0);
}
