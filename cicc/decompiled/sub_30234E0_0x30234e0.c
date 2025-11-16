// Function: sub_30234E0
// Address: 0x30234e0
//
__int64 (*__fastcall sub_30234E0(__int64 a1, __int64 a2))(void)
{
  int v3; // r15d
  __int64 v4; // rdi
  __int64 (*v5)(void); // rax
  char *v6; // r14
  bool v7; // zf
  void (*v8)(); // rax
  __int64 (*result)(void); // rax
  __int64 v10; // r14
  void (*v11)(); // r13
  __int64 v12; // [rsp+8h] [rbp-238h]
  __int64 v13; // [rsp+50h] [rbp-1F0h]
  void (*v14)(); // [rsp+58h] [rbp-1E8h]
  _QWORD *v15; // [rsp+60h] [rbp-1E0h] BYREF
  size_t v16; // [rsp+68h] [rbp-1D8h]
  _BYTE v17[16]; // [rsp+70h] [rbp-1D0h] BYREF
  const char *v18; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v19; // [rsp+88h] [rbp-1B8h]
  __int64 (__fastcall **v20)(); // [rsp+90h] [rbp-1B0h] BYREF
  __int64 (__fastcall **v21)(); // [rsp+98h] [rbp-1A8h] BYREF
  __int64 v22; // [rsp+A0h] [rbp-1A0h]
  __int64 v23; // [rsp+A8h] [rbp-198h]
  unsigned __int64 v24; // [rsp+B0h] [rbp-190h]
  _BYTE *v25; // [rsp+B8h] [rbp-188h]
  unsigned __int64 v26; // [rsp+C0h] [rbp-180h]
  __int64 v27; // [rsp+C8h] [rbp-178h]
  volatile signed __int32 *v28; // [rsp+D0h] [rbp-170h] BYREF
  int v29; // [rsp+D8h] [rbp-168h]
  unsigned __int64 v30[2]; // [rsp+E0h] [rbp-160h] BYREF
  _BYTE v31[16]; // [rsp+F0h] [rbp-150h] BYREF
  _QWORD v32[28]; // [rsp+100h] [rbp-140h] BYREF
  __int16 v33; // [rsp+1E0h] [rbp-60h]
  __int64 v34; // [rsp+1E8h] [rbp-58h]
  __int64 v35; // [rsp+1F0h] [rbp-50h]
  __int64 v36; // [rsp+1F8h] [rbp-48h]
  __int64 v37; // [rsp+200h] [rbp-40h]

  v3 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
  if ( v3 < 0 )
  {
    v10 = *(_QWORD *)(a1 + 224);
    v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
    sub_30232C0((__int64)&v15, a1, v3);
    v18 = "implicit-def: ";
    v20 = (__int64 (__fastcall **)())&v15;
    LOWORD(v22) = 1027;
    if ( v11 != nullsub_98 )
      ((void (__fastcall *)(__int64, const char **, __int64))v11)(v10, &v18, 1);
    if ( v15 != (_QWORD *)v17 )
      j_j___libc_free_0((unsigned __int64)v15);
  }
  else
  {
    v4 = *(_QWORD *)(sub_2E88D60(a2) + 16);
    v12 = *(_QWORD *)(a1 + 224);
    v14 = *(void (**)())(*(_QWORD *)v12 + 120LL);
    v5 = *(__int64 (**)(void))(*(_QWORD *)v4 + 200LL);
    if ( (char *)v5 == (char *)sub_3020000 )
      v13 = v4 + 456;
    else
      v13 = v5();
    sub_222DF20((__int64)v32);
    v33 = 0;
    v32[27] = 0;
    v32[0] = off_4A06798;
    v34 = 0;
    v18 = (const char *)qword_4A072D8;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    *(const char **)((char *)&v18 + qword_4A072D8[-3]) = (const char *)&unk_4A07300;
    v19 = 0;
    sub_222DD70((__int64)&v18 + *((_QWORD *)v18 - 3), 0);
    v20 = (__int64 (__fastcall **)())qword_4A07288;
    *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
    sub_222DD70((__int64)&v20 + (_QWORD)*(v20 - 3), 0);
    v18 = (const char *)qword_4A07328;
    *(const char **)((char *)&v18 + qword_4A07328[-3]) = (const char *)&unk_4A07378;
    v18 = (const char *)off_4A073F0;
    v32[0] = off_4A07440;
    v20 = off_4A07418;
    v22 = 0;
    v23 = 0;
    v21 = off_4A07480;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    sub_220A990(&v28);
    v29 = 24;
    v31[0] = 0;
    v21 = off_4A07080;
    v30[0] = (unsigned __int64)v31;
    v30[1] = 0;
    sub_222DD70((__int64)v32, (__int64)&v21);
    sub_223E0D0((__int64 *)&v20, "reg", 3);
    sub_223E760((__int64 *)&v20, (unsigned int)v3);
    v15 = v17;
    v17[0] = 0;
    v16 = 0;
    if ( v26 )
    {
      if ( v26 <= v24 )
        sub_2241130((unsigned __int64 *)&v15, 0, 0, v25, v24 - (_QWORD)v25);
      else
        sub_2241130((unsigned __int64 *)&v15, 0, 0, v25, v26 - (_QWORD)v25);
    }
    else
    {
      sub_2240AE0((unsigned __int64 *)&v15, v30);
    }
    v6 = sub_C94910(v13 + 432, v15, v16);
    if ( v15 != (_QWORD *)v17 )
      j_j___libc_free_0((unsigned __int64)v15);
    v18 = (const char *)off_4A073F0;
    v32[0] = off_4A07440;
    v20 = off_4A07418;
    v21 = off_4A07080;
    if ( (_BYTE *)v30[0] != v31 )
      j_j___libc_free_0(v30[0]);
    v21 = off_4A07480;
    sub_2209150(&v28);
    v18 = (const char *)qword_4A07328;
    *(const char **)((char *)&v18 + qword_4A07328[-3]) = (const char *)&unk_4A07378;
    v20 = (__int64 (__fastcall **)())qword_4A07288;
    *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
    v18 = (const char *)qword_4A072D8;
    *(const char **)((char *)&v18 + qword_4A072D8[-3]) = (const char *)&unk_4A07300;
    v19 = 0;
    v32[0] = off_4A06798;
    sub_222E050((__int64)v32);
    v7 = *v6 == 0;
    v18 = "implicit-def: ";
    if ( v7 )
    {
      LOWORD(v22) = 259;
      v8 = v14;
      if ( v14 == nullsub_98 )
        goto LABEL_13;
    }
    else
    {
      v8 = v14;
      v20 = (__int64 (__fastcall **)())v6;
      LOWORD(v22) = 771;
      if ( v14 == nullsub_98 )
        goto LABEL_13;
    }
    ((void (__fastcall *)(__int64, const char **, __int64))v8)(v12, &v18, 1);
  }
LABEL_13:
  result = *(__int64 (**)(void))(**(_QWORD **)(a1 + 224) + 160LL);
  if ( (char *)result != (char *)nullsub_99 )
    return (__int64 (*)(void))result();
  return result;
}
