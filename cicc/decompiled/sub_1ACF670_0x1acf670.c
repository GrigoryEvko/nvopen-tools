// Function: sub_1ACF670
// Address: 0x1acf670
//
__int64 __fastcall sub_1ACF670(__int64 a1, const char *a2, int a3, int a4, const char *a5, char a6)
{
  double v6; // xmm0_8
  size_t v7; // rax
  size_t v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  _QWORD *v11; // rsi
  __int64 v12; // rdx
  __int64 v17; // [rsp+40h] [rbp-1D0h]
  char *s; // [rsp+48h] [rbp-1C8h]
  __int64 (__fastcall **v19)(); // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v20; // [rsp+58h] [rbp-1B8h]
  __int64 (__fastcall **v21)(); // [rsp+60h] [rbp-1B0h] BYREF
  _QWORD v22[3]; // [rsp+68h] [rbp-1A8h] BYREF
  unsigned __int64 v23; // [rsp+80h] [rbp-190h]
  __int64 v24; // [rsp+88h] [rbp-188h]
  unsigned __int64 v25; // [rsp+90h] [rbp-180h]
  __int64 v26; // [rsp+98h] [rbp-178h]
  _BYTE v27[8]; // [rsp+A0h] [rbp-170h] BYREF
  int v28; // [rsp+A8h] [rbp-168h]
  _QWORD v29[2]; // [rsp+B0h] [rbp-160h] BYREF
  _QWORD v30[2]; // [rsp+C0h] [rbp-150h] BYREF
  _QWORD v31[28]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v32; // [rsp+1B0h] [rbp-60h]
  __int64 v33; // [rsp+1B8h] [rbp-58h]
  __int64 v34; // [rsp+1C0h] [rbp-50h]
  __int64 v35; // [rsp+1C8h] [rbp-48h]
  __int64 v36; // [rsp+1D0h] [rbp-40h]

  v6 = 0.0;
  if ( a4 )
    v6 = (double)a3 * 100.0 / (double)a4;
  sub_222DF20(v31);
  v31[27] = 0;
  v33 = 0;
  v34 = 0;
  v31[0] = off_4A06798;
  v32 = 0;
  v35 = 0;
  v36 = 0;
  v19 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v19 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v20 = 0;
  sub_222DD70((char *)&v19 + (_QWORD)*(v19 - 3), 0);
  v21 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v22[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v22[-1] + (_QWORD)*(v21 - 3), 0);
  v19 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v19 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v19 = off_4A073F0;
  v31[0] = off_4A07440;
  v21 = off_4A07418;
  v22[1] = 0;
  v22[2] = 0;
  v22[0] = off_4A07480;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_220A990(v27);
  v28 = 24;
  LOBYTE(v30[0]) = 0;
  v22[0] = off_4A07080;
  v29[0] = v30;
  v29[1] = 0;
  sub_222DD70(v31, v22);
  *(_QWORD *)((char *)v22 + (_QWORD)*(v21 - 3)) = 4;
  v7 = strlen(a2);
  sub_223E0D0(&v21, a2, v7);
  sub_223E0D0(&v21, ": ", 2);
  s = (char *)sub_223E730(&v21, (unsigned int)a3);
  sub_223E0D0(s, " [", 2);
  v17 = sub_223EB60(s, v6);
  sub_223E0D0(v17, "% of ", 5);
  v8 = strlen(a5);
  sub_223E0D0(v17, a5, v8);
  sub_223E0D0(v17, "]", 1);
  if ( a6 )
    sub_223E0D0(&v21, "\n", 1);
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v9 = v25;
  *(_QWORD *)(a1 + 8) = 0;
  if ( v9 )
  {
    if ( v9 > v23 )
      v10 = v9 - v24;
    else
      v10 = v23 - v24;
    v11 = 0;
    sub_2241130(a1, 0, 0, v24, v10);
  }
  else
  {
    v11 = v29;
    sub_2240AE0(a1, v29);
  }
  v19 = off_4A073F0;
  v31[0] = off_4A07440;
  v21 = off_4A07418;
  v22[0] = off_4A07080;
  if ( (_QWORD *)v29[0] != v30 )
  {
    v11 = (_QWORD *)(v30[0] + 1LL);
    j_j___libc_free_0(v29[0], v30[0] + 1LL);
  }
  v22[0] = off_4A07480;
  sub_2209150(v27, v11, v12);
  v19 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v19 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v21 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v22[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v19 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v19 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v20 = 0;
  v31[0] = off_4A06798;
  sub_222E050(v31);
  return a1;
}
