// Function: sub_23C8E10
// Address: 0x23c8e10
//
__int64 __fastcall sub_23C8E10(__int64 a1, __int64 a2, _BYTE *a3, size_t a4, __int32 a5, char a6)
{
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r11
  char v13; // dl
  char v14; // al
  unsigned __int64 v16; // rax
  __int64 (__fastcall **v17)(); // r13
  __int64 *v18; // rsi
  __int64 v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-190h]
  __int64 v23; // [rsp+18h] [rbp-178h] BYREF
  unsigned __int64 v24; // [rsp+20h] [rbp-170h] BYREF
  __int64 v25; // [rsp+28h] [rbp-168h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-160h] BYREF
  __int64 v27; // [rsp+40h] [rbp-150h] BYREF
  __int64 v28[2]; // [rsp+50h] [rbp-140h] BYREF
  __int64 v29; // [rsp+60h] [rbp-130h] BYREF
  const char *v30; // [rsp+70h] [rbp-120h] BYREF
  __int32 v31; // [rsp+80h] [rbp-110h]
  __int16 v32; // [rsp+90h] [rbp-100h]
  _QWORD v33[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v34; // [rsp+C0h] [rbp-D0h]
  _QWORD v35[4]; // [rsp+D0h] [rbp-C0h] BYREF
  __int16 v36; // [rsp+F0h] [rbp-A0h]
  _QWORD v37[4]; // [rsp+100h] [rbp-90h] BYREF
  __int16 v38; // [rsp+120h] [rbp-70h]
  void *v39[4]; // [rsp+130h] [rbp-60h] BYREF
  __int16 v40; // [rsp+150h] [rbp-40h]

  v10 = sub_C92610();
  v11 = *sub_23C78F0(a2, a3, a4, v10);
  v12 = v11 + 8;
  if ( v13
    && (v21 = v11 + 8,
        sub_23C7DD0(&v23, *(_QWORD *)(v11 + 8), a3, a4, a5, a6),
        v12 = v21,
        (v23 & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    v16 = v23 & 0xFFFFFFFFFFFFFFFELL | 1;
    v23 = 0;
    v24 = v16;
    sub_C64870((__int64)v26, (__int64 *)&v24);
    v33[0] = &v30;
    v31 = a5;
    v32 = 2307;
    v33[2] = ": '";
    v34 = 770;
    v35[0] = v33;
    v37[2] = "': ";
    v30 = "malformed section at line ";
    v37[0] = v35;
    v36 = 1282;
    v39[0] = v37;
    v35[2] = a3;
    v35[3] = a4;
    v38 = 770;
    v40 = 1026;
    v39[2] = v26;
    v17 = sub_2241E50();
    sub_CA0F50(v28, v39);
    v18 = v28;
    sub_C63F00(&v25, (__int64)v28, 0x16u, (__int64)v17);
    if ( (__int64 *)v28[0] != &v29 )
    {
      v18 = (__int64 *)(v29 + 1);
      j_j___libc_free_0(v28[0]);
    }
    v19 = v25;
    v20 = (__int64 *)v26[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v19 & 0xFFFFFFFFFFFFFFFELL;
    if ( v20 != &v27 )
    {
      v18 = (__int64 *)(v27 + 1);
      j_j___libc_free_0((unsigned __int64)v20);
    }
    if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v24, (__int64)v18);
    if ( (v23 & 1) != 0 || (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v23, (__int64)v18);
  }
  else
  {
    v14 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = v12;
    *(_BYTE *)(a1 + 8) = v14 & 0xFC | 2;
  }
  return a1;
}
