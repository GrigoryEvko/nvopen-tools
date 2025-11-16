// Function: sub_36FBAC0
// Address: 0x36fbac0
//
__int64 __fastcall sub_36FBAC0(__int64 a1, const char *a2, signed int a3, int a4, const char *a5, char a6)
{
  double v6; // xmm0_8
  size_t v7; // rax
  size_t v8; // rax
  unsigned __int64 v9; // rax
  size_t v10; // r8
  __int64 *v15; // [rsp+40h] [rbp-1D0h]
  __int64 *s; // [rsp+48h] [rbp-1C8h]
  __int64 (__fastcall **v17)(); // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v18; // [rsp+58h] [rbp-1B8h]
  __int64 (__fastcall **v19)(); // [rsp+60h] [rbp-1B0h] BYREF
  _QWORD v20[3]; // [rsp+68h] [rbp-1A8h] BYREF
  unsigned __int64 v21; // [rsp+80h] [rbp-190h]
  _BYTE *v22; // [rsp+88h] [rbp-188h]
  unsigned __int64 v23; // [rsp+90h] [rbp-180h]
  __int64 v24; // [rsp+98h] [rbp-178h]
  volatile signed __int32 *v25; // [rsp+A0h] [rbp-170h] BYREF
  int v26; // [rsp+A8h] [rbp-168h]
  unsigned __int64 v27[2]; // [rsp+B0h] [rbp-160h] BYREF
  _BYTE v28[16]; // [rsp+C0h] [rbp-150h] BYREF
  _QWORD v29[28]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v30; // [rsp+1B0h] [rbp-60h]
  __int64 v31; // [rsp+1B8h] [rbp-58h]
  __int64 v32; // [rsp+1C0h] [rbp-50h]
  __int64 v33; // [rsp+1C8h] [rbp-48h]
  __int64 v34; // [rsp+1D0h] [rbp-40h]

  v6 = 0.0;
  if ( a4 )
    v6 = (double)a3 * 100.0 / (double)a4;
  sub_222DF20((__int64)v29);
  v29[27] = 0;
  v31 = 0;
  v32 = 0;
  v29[0] = off_4A06798;
  v30 = 0;
  v33 = 0;
  v34 = 0;
  v17 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v17 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v18 = 0;
  sub_222DD70((__int64)&v17 + (_QWORD)*(v17 - 3), 0);
  v19 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v20[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((__int64)&v20[-1] + (_QWORD)*(v19 - 3), 0);
  v17 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v17 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v17 = off_4A073F0;
  v29[0] = off_4A07440;
  v19 = off_4A07418;
  v20[1] = 0;
  v20[2] = 0;
  v20[0] = off_4A07480;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  sub_220A990(&v25);
  v26 = 24;
  v28[0] = 0;
  v20[0] = off_4A07080;
  v27[0] = (unsigned __int64)v28;
  v27[1] = 0;
  sub_222DD70((__int64)v29, (__int64)v20);
  *(_QWORD *)((char *)v20 + (_QWORD)*(v19 - 3)) = 4;
  v7 = strlen(a2);
  sub_223E0D0((__int64 *)&v19, a2, v7);
  sub_223E0D0((__int64 *)&v19, ": ", 2);
  s = sub_223E730((__int64 *)&v19, a3);
  sub_223E0D0(s, " [", 2);
  v15 = sub_223EB60(s, v6);
  sub_223E0D0(v15, "% of ", 5);
  v8 = strlen(a5);
  sub_223E0D0(v15, a5, v8);
  sub_223E0D0(v15, "]", 1);
  if ( a6 )
    sub_223E0D0((__int64 *)&v19, "\n", 1);
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v9 = v23;
  *(_QWORD *)(a1 + 8) = 0;
  if ( v9 )
  {
    if ( v9 > v21 )
      v10 = v9 - (_QWORD)v22;
    else
      v10 = v21 - (_QWORD)v22;
    sub_2241130((unsigned __int64 *)a1, 0, 0, v22, v10);
  }
  else
  {
    sub_2240AE0((unsigned __int64 *)a1, v27);
  }
  v17 = off_4A073F0;
  v29[0] = off_4A07440;
  v19 = off_4A07418;
  v20[0] = off_4A07080;
  if ( (_BYTE *)v27[0] != v28 )
    j_j___libc_free_0(v27[0]);
  v20[0] = off_4A07480;
  sub_2209150(&v25);
  v17 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v17 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v19 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v20[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v17 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v17 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v18 = 0;
  v29[0] = off_4A06798;
  sub_222E050((__int64)v29);
  return a1;
}
