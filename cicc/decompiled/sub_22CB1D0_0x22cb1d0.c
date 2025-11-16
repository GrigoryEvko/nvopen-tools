// Function: sub_22CB1D0
// Address: 0x22cb1d0
//
__int64 __fastcall sub_22CB1D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v9; // rdx
  _BYTE *v10; // r14
  _BYTE *v11; // rsi
  unsigned __int64 v12; // r10
  __int64 v13; // rax
  unsigned __int64 *v15; // rsi
  __int64 *v16; // rdi
  __int64 v17; // rdx
  bool v18; // zf
  unsigned __int64 v20; // [rsp+10h] [rbp-100h]
  __int64 v21; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v22[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v23[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v24; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v25; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v26; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v27; // [rsp+60h] [rbp-B0h]
  unsigned int v28; // [rsp+68h] [rbp-A8h]
  char v29; // [rsp+70h] [rbp-A0h]
  __int64 v30[2]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v31; // [rsp+90h] [rbp-80h] BYREF
  char v32; // [rsp+A0h] [rbp-70h]
  unsigned __int8 v33[40]; // [rsp+B0h] [rbp-60h] BYREF
  char v34; // [rsp+D8h] [rbp-38h]

  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(a3 - 8);
  else
    v9 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v10 = *(_BYTE **)v9;
  v11 = *(_BYTE **)(v9 + 32);
  v22[0] = a2;
  v21 = (__int64)v11;
  v20 = a2;
  v22[1] = a5;
  sub_22C7770((__int64)&v25, a2, (__int64)v10, a3, a4);
  v12 = v20;
  v13 = a4;
  if ( v29 )
  {
    if ( *v11 == 86
      && (sub_22CAB90((__int64)v33, v22, (__int64)v10, (__int64)&v25, (__int64)v11, 1), v12 = v20, v13 = a4, v34) )
    {
      sub_22C05A0(a1, v33);
      v18 = v34 == 0;
      *(_BYTE *)(a1 + 40) = 1;
      if ( !v18 )
      {
        v34 = 0;
        sub_22C0090(v33);
      }
    }
    else
    {
      v15 = (unsigned __int64 *)v12;
      v16 = v30;
      sub_22C7770((__int64)v30, v12, v21, a3, v13);
      if ( v32 )
      {
        if ( *v10 == 86
          && (v15 = v22, v16 = (__int64 *)v33, sub_22CAB90((__int64)v33, v22, v21, (__int64)v30, (__int64)v10, 0), v34) )
        {
          sub_22C05A0(a1, v33);
          v18 = v34 == 0;
          *(_BYTE *)(a1 + 40) = 1;
          if ( !v18 )
          {
            v34 = 0;
            sub_22C0090(v33);
          }
        }
        else
        {
          if ( !*(_QWORD *)(a5 + 16) )
            sub_4263D6(v16, v15, v17);
          (*(void (__fastcall **)(__int64 *, unsigned __int64, unsigned __int64 *, __int64 *))(a5 + 24))(
            v23,
            a5,
            &v25,
            v30);
          sub_22C06B0((__int64)v33, (__int64)v23, 0);
          sub_22C0650(a1, v33);
          *(_BYTE *)(a1 + 40) = 1;
          sub_22C0090(v33);
          sub_969240(&v24);
          sub_969240(v23);
        }
        if ( v32 )
        {
          v32 = 0;
          sub_969240(&v31);
          sub_969240(v30);
        }
      }
      else
      {
        *(_BYTE *)(a1 + 40) = 0;
      }
    }
    if ( v29 )
    {
      v29 = 0;
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 0;
  }
  return a1;
}
