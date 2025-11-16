// Function: sub_2BDCD80
// Address: 0x2bdcd80
//
__int64 __fastcall sub_2BDCD80(char a1, int a2)
{
  char *v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r14d
  _BYTE *v6[2]; // [rsp+30h] [rbp-1D0h] BYREF
  _QWORD v7[2]; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v8)(); // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v9; // [rsp+58h] [rbp-1A8h]
  _QWORD v10[7]; // [rsp+60h] [rbp-1A0h] BYREF
  volatile signed __int32 *v11; // [rsp+98h] [rbp-168h] BYREF
  int v12; // [rsp+A0h] [rbp-160h]
  _QWORD *v13; // [rsp+A8h] [rbp-158h] BYREF
  _QWORD v14[2]; // [rsp+B8h] [rbp-148h] BYREF
  _QWORD v15[4]; // [rsp+C8h] [rbp-138h] BYREF
  char v16; // [rsp+E8h] [rbp-118h]
  __int64 v17; // [rsp+1A0h] [rbp-60h]
  __int16 v18; // [rsp+1A8h] [rbp-58h]
  __int64 v19; // [rsp+1B0h] [rbp-50h]
  __int64 v20; // [rsp+1B8h] [rbp-48h]
  __int64 v21; // [rsp+1C0h] [rbp-40h]
  __int64 v22; // [rsp+1C8h] [rbp-38h]

  v6[0] = v7;
  sub_2240A50((__int64 *)v6, 1u, a1);
  sub_222DF20((__int64)v15);
  v17 = 0;
  v19 = 0;
  v20 = 0;
  v15[0] = off_4A06798;
  v18 = 0;
  v21 = 0;
  v22 = 0;
  v8 = (__int64 (__fastcall **)())qword_4A07108;
  *(__int64 (__fastcall ***)())((char *)&v8 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
  v9 = 0;
  sub_222DD70((__int64)&v8 + (_QWORD)*(v8 - 3), 0);
  memset(&v10[1], 0, 48);
  v8 = off_4A07178;
  v15[0] = off_4A071A0;
  v10[0] = off_4A07480;
  sub_220A990(&v11);
  v12 = 0;
  v10[0] = off_4A07080;
  v13 = v14;
  sub_2BDC240((__int64 *)&v13, v6[0], (__int64)&v6[0][(unsigned __int64)v6[1]]);
  v12 = 8;
  sub_223FD50((__int64)v10, (__int64)v13, 0, 0);
  sub_222DD70((__int64)v15, (__int64)v10);
  if ( (_QWORD *)v6[0] != v7 )
    j_j___libc_free_0((unsigned __int64)v6[0]);
  if ( a2 == 8 )
  {
    v2 = (char *)&v8 + (_QWORD)*(v8 - 3);
    *((_DWORD *)v2 + 6) = *((_DWORD *)v2 + 6) & 0xFFFFFFB5 | 0x40;
  }
  else if ( a2 == 16 )
  {
    v2 = (char *)&v8 + (_QWORD)*(v8 - 3);
    *((_DWORD *)v2 + 6) = *((_DWORD *)v2 + 6) & 0xFFFFFFB5 | 8;
  }
  sub_222E730((__int64 *)&v8, (__int64)v6, (__int64)v2, v3);
  if ( (v16 & 5) != 0 )
    v4 = -1;
  else
    v4 = (unsigned int)v6[0];
  v8 = off_4A07178;
  v15[0] = off_4A071A0;
  v10[0] = off_4A07080;
  if ( v13 != v14 )
    j_j___libc_free_0((unsigned __int64)v13);
  v10[0] = off_4A07480;
  sub_2209150(&v11);
  v8 = (__int64 (__fastcall **)())qword_4A07108;
  *(__int64 (__fastcall ***)())((char *)&v8 + qword_4A07108[-3]) = (__int64 (__fastcall **)())&unk_4A07130;
  v9 = 0;
  v15[0] = off_4A06798;
  sub_222E050((__int64)v15);
  return v4;
}
