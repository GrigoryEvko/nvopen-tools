// Function: sub_127AC30
// Address: 0x127ac30
//
__int64 __fastcall sub_127AC30(__int64 a1, _QWORD *a2, __int64 a3, char a4)
{
  unsigned __int64 v4; // rax
  _QWORD *v5; // rsi
  __int64 v6; // rdx
  size_t v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  size_t v11; // rax
  _QWORD *v12; // r8
  _QWORD *v14; // [rsp+8h] [rbp-218h]
  char *v16; // [rsp+10h] [rbp-210h]
  char *v17; // [rsp+10h] [rbp-210h]
  _QWORD *v18; // [rsp+10h] [rbp-210h]
  char v19; // [rsp+5Fh] [rbp-1C1h] BYREF
  __int64 (__fastcall **v20)(); // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v21; // [rsp+68h] [rbp-1B8h]
  __int64 (__fastcall **v22)(); // [rsp+70h] [rbp-1B0h] BYREF
  _QWORD v23[3]; // [rsp+78h] [rbp-1A8h] BYREF
  unsigned __int64 v24; // [rsp+90h] [rbp-190h]
  __int64 v25; // [rsp+98h] [rbp-188h]
  unsigned __int64 v26; // [rsp+A0h] [rbp-180h]
  __int64 v27; // [rsp+A8h] [rbp-178h]
  char v28[8]; // [rsp+B0h] [rbp-170h] BYREF
  int v29; // [rsp+B8h] [rbp-168h]
  _QWORD v30[2]; // [rsp+C0h] [rbp-160h] BYREF
  _QWORD v31[2]; // [rsp+D0h] [rbp-150h] BYREF
  _QWORD v32[28]; // [rsp+E0h] [rbp-140h] BYREF
  __int16 v33; // [rsp+1C0h] [rbp-60h]
  __int64 v34; // [rsp+1C8h] [rbp-58h]
  __int64 v35; // [rsp+1D0h] [rbp-50h]
  __int64 v36; // [rsp+1D8h] [rbp-48h]
  __int64 v37; // [rsp+1E0h] [rbp-40h]

  sub_222DF20(v32);
  v32[27] = 0;
  v34 = 0;
  v35 = 0;
  v32[0] = off_4A06798;
  v33 = 0;
  v36 = 0;
  v37 = 0;
  v20 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v21 = 0;
  sub_222DD70((char *)&v20 + (_QWORD)*(v20 - 3), 0);
  v22 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v23[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v23[-1] + (_QWORD)*(v22 - 3), 0);
  v20 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v23[1] = 0;
  v23[2] = 0;
  v20 = off_4A073F0;
  v24 = 0;
  v25 = 0;
  v32[0] = off_4A07440;
  v26 = 0;
  v27 = 0;
  v22 = off_4A07418;
  v23[0] = off_4A07480;
  sub_220A990(v28);
  v29 = 24;
  LOBYTE(v31[0]) = 0;
  v23[0] = off_4A07080;
  v30[0] = v31;
  v30[1] = 0;
  sub_222DD70(v32, v23);
  if ( !byte_4F92CE8 && (unsigned int)sub_2207590(&byte_4F92CE8) )
  {
    qword_4F92CF0 = sub_723F40(0);
    sub_2207640(&byte_4F92CE8);
  }
  if ( !byte_4F92CD8 && (unsigned int)sub_2207590(&byte_4F92CD8) )
  {
    dword_4F92CE0 = strlen(qword_4F92CF0);
    sub_2207640(&byte_4F92CD8);
  }
  if ( qword_4D045BC && *(_QWORD *)(a3 + 8) && (a4 || (*(_BYTE *)(a3 + 88) & 0x70) == 0x10) )
  {
    if ( off_4B7D3F8 )
    {
      v16 = off_4B7D3F8;
      v8 = strlen(off_4B7D3F8);
      sub_223E0D0(&v22, v16, v8);
    }
    else
    {
      sub_222DC80((char *)&v23[-1] + (_QWORD)*(v22 - 3), *(_DWORD *)((char *)&v23[3] + (_QWORD)*(v22 - 3)) | 1u);
    }
    v9 = sub_223E730(&v22, (unsigned int)dword_4F92CE0);
    v19 = 95;
    v10 = (_QWORD *)sub_223E0D0(v9, &v19, 1);
    if ( qword_4F92CF0 )
    {
      v17 = qword_4F92CF0;
      v14 = v10;
      v11 = strlen(qword_4F92CF0);
      sub_223E0D0(v14, v17, v11);
      v12 = v14;
    }
    else
    {
      v18 = v10;
      sub_222DC80((char *)v10 + *(_QWORD *)(*v10 - 24LL), *(_DWORD *)((char *)v10 + *(_QWORD *)(*v10 - 24LL) + 32) | 1u);
      v12 = v18;
    }
    v19 = 95;
    sub_223E0D0(v12, &v19, 1);
  }
  sub_223E0D0(&v22, *a2, a2[1]);
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v4 = v26;
  *(_QWORD *)(a1 + 8) = 0;
  if ( v4 )
  {
    v5 = 0;
    if ( v4 > v24 )
      sub_2241130(a1, 0, 0, v25, v4 - v25);
    else
      sub_2241130(a1, 0, 0, v25, v24 - v25);
  }
  else
  {
    v5 = v30;
    sub_2240AE0(a1, v30);
  }
  v20 = off_4A073F0;
  v32[0] = off_4A07440;
  v22 = off_4A07418;
  v23[0] = off_4A07080;
  if ( (_QWORD *)v30[0] != v31 )
  {
    v5 = (_QWORD *)(v31[0] + 1LL);
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  }
  v23[0] = off_4A07480;
  sub_2209150(v28, v5, v6);
  v20 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v22 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v23[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v20 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v20 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v21 = 0;
  v32[0] = off_4A06798;
  sub_222E050(v32);
  return a1;
}
