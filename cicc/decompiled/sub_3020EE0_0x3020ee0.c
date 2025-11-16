// Function: sub_3020EE0
// Address: 0x3020ee0
//
__int64 __fastcall sub_3020EE0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r12
  __int64 v5; // rdx
  const char *v6; // rsi
  __int64 *v7; // rax
  __int64 *v8; // r9
  size_t v9; // r8
  __int64 *v10; // r9
  __int64 *v12; // rsi
  __int64 v13; // rdx
  __int64 *v14; // [rsp+0h] [rbp-260h]
  __int64 *v15; // [rsp+0h] [rbp-260h]
  unsigned __int64 v16[2]; // [rsp+50h] [rbp-210h] BYREF
  _BYTE v17[16]; // [rsp+60h] [rbp-200h] BYREF
  unsigned __int64 *v18; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v19; // [rsp+78h] [rbp-1E8h]
  _BYTE v20[16]; // [rsp+80h] [rbp-1E0h] BYREF
  __int16 v21; // [rsp+90h] [rbp-1D0h]
  __int64 (__fastcall **v22)(); // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v23; // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v24)(); // [rsp+B0h] [rbp-1B0h] BYREF
  _QWORD v25[3]; // [rsp+B8h] [rbp-1A8h] BYREF
  unsigned __int64 v26; // [rsp+D0h] [rbp-190h]
  _BYTE *v27; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v28; // [rsp+E0h] [rbp-180h]
  __int64 v29; // [rsp+E8h] [rbp-178h]
  volatile signed __int32 *v30; // [rsp+F0h] [rbp-170h] BYREF
  int v31; // [rsp+F8h] [rbp-168h]
  unsigned __int64 v32[2]; // [rsp+100h] [rbp-160h] BYREF
  _BYTE v33[16]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v34[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v35; // [rsp+200h] [rbp-60h]
  __int64 v36; // [rsp+208h] [rbp-58h]
  __int64 v37; // [rsp+210h] [rbp-50h]
  __int64 v38; // [rsp+218h] [rbp-48h]
  __int64 v39; // [rsp+220h] [rbp-40h]

  sub_222DF20((__int64)v34);
  v35 = 0;
  v34[27] = 0;
  v34[0] = off_4A06798;
  v36 = 0;
  v22 = (__int64 (__fastcall **)())qword_4A072D8;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  *(__int64 (__fastcall ***)())((char *)&v22 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v23 = 0;
  sub_222DD70((__int64)&v22 + (_QWORD)*(v22 - 3), 0);
  v24 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v25[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((__int64)&v25[-1] + (_QWORD)*(v24 - 3), 0);
  v22 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v22 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v22 = off_4A073F0;
  v34[0] = off_4A07440;
  v24 = off_4A07418;
  v25[0] = off_4A07480;
  v25[1] = 0;
  v25[2] = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  sub_220A990(&v30);
  v31 = 24;
  v33[0] = 0;
  v25[0] = off_4A07080;
  v32[0] = (unsigned __int64)v33;
  v32[1] = 0;
  sub_222DD70((__int64)v34, (__int64)v25);
  v4 = sub_E6C430(*(_QWORD *)(a1 + 216), (__int64)v25, v1, v2, v3);
  sub_223E0D0((__int64 *)&v24, "\tbra.uni\t", 9);
  if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
  {
    v12 = *(__int64 **)(v4 - 8);
    v13 = *v12;
    v18 = (unsigned __int64 *)v20;
    sub_3020610((__int64 *)&v18, (_BYTE *)v12 + 24, (__int64)v12 + v13 + 24);
    v5 = v19;
    v6 = (const char *)v18;
  }
  else
  {
    v20[0] = 0;
    v5 = 0;
    v18 = (unsigned __int64 *)v20;
    v6 = v20;
    v19 = 0;
  }
  v7 = sub_223E0D0((__int64 *)&v24, v6, v5);
  sub_223E0D0(v7, ";\n", 2);
  if ( v18 != (unsigned __int64 *)v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  v16[1] = 0;
  v16[0] = (unsigned __int64)v17;
  v8 = *(__int64 **)(a1 + 224);
  v17[0] = 0;
  if ( v28 )
  {
    v14 = v8;
    if ( v28 > v26 )
      v9 = v28 - (_QWORD)v27;
    else
      v9 = v26 - (_QWORD)v27;
    sub_2241130(v16, 0, 0, v27, v9);
    v10 = v14;
  }
  else
  {
    v15 = v8;
    sub_2240AE0(v16, v32);
    v10 = v15;
  }
  v18 = v16;
  v21 = 260;
  sub_E99A90(v10, (__int64)&v18);
  if ( (_BYTE *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v4, 0);
  v22 = off_4A073F0;
  v34[0] = off_4A07440;
  v24 = off_4A07418;
  v25[0] = off_4A07080;
  if ( (_BYTE *)v32[0] != v33 )
    j_j___libc_free_0(v32[0]);
  v25[0] = off_4A07480;
  sub_2209150(&v30);
  v22 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v22 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v24 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v25[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v22 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v22 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v23 = 0;
  v34[0] = off_4A06798;
  sub_222E050((__int64)v34);
  return v4;
}
