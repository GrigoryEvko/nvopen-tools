// Function: sub_215BB80
// Address: 0x215bb80
//
__int64 *__fastcall sub_215BB80(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // rax
  __int64 v3; // r9
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // [rsp+8h] [rbp-248h]
  __int64 v12; // [rsp+8h] [rbp-248h]
  const char *v13; // [rsp+50h] [rbp-200h] BYREF
  __int16 v14; // [rsp+60h] [rbp-1F0h]
  const char *v15; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v16; // [rsp+78h] [rbp-1D8h]
  char v17[16]; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 (__fastcall **v18)(); // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v19; // [rsp+98h] [rbp-1B8h]
  __int64 (__fastcall **v20)(); // [rsp+A0h] [rbp-1B0h] BYREF
  _QWORD v21[3]; // [rsp+A8h] [rbp-1A8h] BYREF
  unsigned __int64 v22; // [rsp+C0h] [rbp-190h]
  __int64 v23; // [rsp+C8h] [rbp-188h]
  unsigned __int64 v24; // [rsp+D0h] [rbp-180h]
  __int64 v25; // [rsp+D8h] [rbp-178h]
  _BYTE v26[8]; // [rsp+E0h] [rbp-170h] BYREF
  int v27; // [rsp+E8h] [rbp-168h]
  _QWORD v28[2]; // [rsp+F0h] [rbp-160h] BYREF
  _QWORD v29[2]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD v30[28]; // [rsp+110h] [rbp-140h] BYREF
  __int16 v31; // [rsp+1F0h] [rbp-60h]
  __int64 v32; // [rsp+1F8h] [rbp-58h]
  __int64 v33; // [rsp+200h] [rbp-50h]
  __int64 v34; // [rsp+208h] [rbp-48h]
  __int64 v35; // [rsp+210h] [rbp-40h]

  sub_222DF20(v30);
  v31 = 0;
  v30[27] = 0;
  v30[0] = off_4A06798;
  v32 = 0;
  v18 = (__int64 (__fastcall **)())qword_4A072D8;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  *(__int64 (__fastcall ***)())((char *)&v18 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v19 = 0;
  sub_222DD70((char *)&v18 + (_QWORD)*(v18 - 3), 0);
  v20 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v21[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v21[-1] + (_QWORD)*(v20 - 3), 0);
  v18 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v18 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v21[1] = 0;
  v21[2] = 0;
  v18 = off_4A073F0;
  v22 = 0;
  v23 = 0;
  v30[0] = off_4A07440;
  v24 = 0;
  v25 = 0;
  v20 = off_4A07418;
  v21[0] = off_4A07480;
  sub_220A990(v26);
  v27 = 24;
  LOBYTE(v29[0]) = 0;
  v21[0] = off_4A07080;
  v28[0] = v29;
  v28[1] = 0;
  sub_222DD70(v30, v21);
  v1 = (__int64 *)sub_38BFA60(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL), 1);
  sub_223E0D0(&v20, "\tbra.uni\t", 9);
  if ( ((*v1 >> 2) & 1) != 0 )
  {
    v9 = (__int64 *)*(v1 - 1);
    v10 = *v9;
    v15 = v17;
    sub_215B830((__int64 *)&v15, (_BYTE *)v9 + 16, (__int64)v9 + v10 + 16);
    v2 = sub_223E0D0(&v20, v15);
  }
  else
  {
    v16 = 0;
    v15 = v17;
    v17[0] = 0;
    v2 = sub_223E0D0(&v20, v17);
  }
  sub_223E0D0(v2, ";\n", 2);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, *(_QWORD *)v17 + 1LL);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v15 = v17;
  v16 = 0;
  v17[0] = 0;
  if ( v24 )
  {
    v11 = v3;
    if ( v24 > v22 )
      v4 = v24 - v23;
    else
      v4 = v22 - v23;
    sub_2241130(&v15, 0, 0, v23, v4);
    v5 = v11;
  }
  else
  {
    v12 = v3;
    sub_2240AE0(&v15, v28);
    v5 = v12;
  }
  v14 = 257;
  if ( *v15 )
  {
    v13 = v15;
    LOBYTE(v14) = 3;
  }
  sub_38DD5A0(v5, &v13);
  if ( v15 != v17 )
    j_j___libc_free_0(v15, *(_QWORD *)v17 + 1LL);
  v6 = (__int64)v1;
  (*(void (__fastcall **)(_QWORD, __int64 *, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
    v1,
    0);
  v18 = off_4A073F0;
  v30[0] = off_4A07440;
  v20 = off_4A07418;
  v21[0] = off_4A07080;
  if ( (_QWORD *)v28[0] != v29 )
  {
    v6 = v29[0] + 1LL;
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  }
  v21[0] = off_4A07480;
  sub_2209150(v26, v6, v7);
  v18 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v18 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v20 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v21[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v18 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v18 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v19 = 0;
  v30[0] = off_4A06798;
  sub_222E050(v30);
  return v1;
}
