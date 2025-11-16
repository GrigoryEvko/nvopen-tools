// Function: sub_2640980
// Address: 0x2640980
//
__int64 __fastcall sub_2640980(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  size_t v3; // r8
  __int64 (__fastcall **v5)(); // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v6; // [rsp+48h] [rbp-1B8h]
  __int64 (__fastcall **v7)(); // [rsp+50h] [rbp-1B0h] BYREF
  _QWORD v8[2]; // [rsp+58h] [rbp-1A8h] BYREF
  __int64 v9; // [rsp+68h] [rbp-198h]
  unsigned __int64 v10; // [rsp+70h] [rbp-190h]
  _BYTE *v11; // [rsp+78h] [rbp-188h]
  unsigned __int64 v12; // [rsp+80h] [rbp-180h]
  __int64 v13; // [rsp+88h] [rbp-178h]
  volatile signed __int32 *v14; // [rsp+90h] [rbp-170h] BYREF
  int v15; // [rsp+98h] [rbp-168h]
  unsigned __int64 v16[2]; // [rsp+A0h] [rbp-160h] BYREF
  char v17; // [rsp+B0h] [rbp-150h] BYREF
  _QWORD v18[28]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v19; // [rsp+1A0h] [rbp-60h]
  __int64 v20; // [rsp+1A8h] [rbp-58h]
  __int64 v21; // [rsp+1B0h] [rbp-50h]
  __int64 v22; // [rsp+1B8h] [rbp-48h]
  __int64 v23; // [rsp+1C0h] [rbp-40h]

  sub_222DF20((__int64)v18);
  v18[27] = 0;
  v20 = 0;
  v18[0] = off_4A06798;
  v19 = 0;
  v5 = (__int64 (__fastcall **)())qword_4A072D8;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  *(__int64 (__fastcall ***)())((char *)&v5 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v6 = 0;
  sub_222DD70((__int64)&v5 + (_QWORD)*(v5 - 3), 0);
  v7 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v8[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((__int64)&v8[-1] + (_QWORD)*(v7 - 3), 0);
  v5 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v5 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v5 = off_4A073F0;
  v18[0] = off_4A07440;
  v7 = off_4A07418;
  v8[1] = 0;
  v8[0] = off_4A07480;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  sub_220A990(&v14);
  v15 = 24;
  v17 = 0;
  v8[0] = off_4A07080;
  v16[0] = (unsigned __int64)&v17;
  v16[1] = 0;
  sub_222DD70((__int64)v18, (__int64)v8);
  *(_DWORD *)((char *)&v9 + (_QWORD)*(v7 - 3)) = *(_DWORD *)((_BYTE *)&v9 + (_QWORD)*(v7 - 3)) & 0xFFFFFFB5 | 8;
  sub_223E0D0((__int64 *)&v7, "N0x", 3);
  sub_223E960((__int64 *)&v7, a2);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v2 = v12;
  *(_BYTE *)(a1 + 16) = 0;
  if ( v2 )
  {
    if ( v2 > v10 )
      v3 = v2 - (_QWORD)v11;
    else
      v3 = v10 - (_QWORD)v11;
    sub_2241130((unsigned __int64 *)a1, 0, 0, v11, v3);
  }
  else
  {
    sub_2240AE0((unsigned __int64 *)a1, v16);
  }
  v5 = off_4A073F0;
  v18[0] = off_4A07440;
  v7 = off_4A07418;
  v8[0] = off_4A07080;
  sub_2240A30(v16);
  v8[0] = off_4A07480;
  sub_2209150(&v14);
  v5 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v5 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v7 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v8[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v5 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v5 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v6 = 0;
  v18[0] = off_4A06798;
  sub_222E050((__int64)v18);
  return a1;
}
