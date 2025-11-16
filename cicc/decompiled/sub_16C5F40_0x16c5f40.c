// Function: sub_16C5F40
// Address: 0x16c5f40
//
__int64 __fastcall sub_16C5F40(__int64 a1, int *a2, __int64 a3, char a4, unsigned int a5, int a6, char a7)
{
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r15
  _BYTE *v13; // rax
  int *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned int v26; // r15d
  int v27; // [rsp+8h] [rbp-1F8h]
  unsigned int v28; // [rsp+10h] [rbp-1F0h]
  __int64 v29; // [rsp+10h] [rbp-1F0h]
  _BYTE **v32; // [rsp+30h] [rbp-1D0h] BYREF
  __int16 v33; // [rsp+40h] [rbp-1C0h]
  char v34[16]; // [rsp+50h] [rbp-1B0h] BYREF
  __int16 v35; // [rsp+60h] [rbp-1A0h]
  char v36[16]; // [rsp+70h] [rbp-190h] BYREF
  __int16 v37; // [rsp+80h] [rbp-180h]
  char v38[16]; // [rsp+90h] [rbp-170h] BYREF
  __int16 v39; // [rsp+A0h] [rbp-160h]
  _BYTE *v40; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-148h]
  _BYTE v42[128]; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v43[2]; // [rsp+140h] [rbp-C0h] BYREF
  _WORD v44[88]; // [rsp+150h] [rbp-B0h] BYREF

  v40 = v42;
  v41 = 0x8000000000LL;
  sub_16E2F40(a1, &v40);
  if ( a4 )
  {
    v43[0] = (__int64)&v40;
    v44[0] = 262;
    if ( !(unsigned __int8)sub_16C4E60((__int64)v43, 2u) )
    {
      v43[1] = 0x8000000000LL;
      v43[0] = (__int64)v44;
      sub_16C5C30(1, (__int64)v43);
      v39 = 257;
      v37 = 257;
      v35 = 257;
      v33 = 262;
      v32 = &v40;
      sub_16C4D40((__int64)v43, (__int64)&v32, (__int64)v34, (__int64)v36, (__int64)v38);
      sub_16C5D80((__int64 *)&v40, v43);
      if ( (_WORD *)v43[0] != v44 )
        _libc_free(v43[0]);
    }
  }
  sub_16C3310(a3, (__int64)&v40);
  v10 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v10 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 1);
    v10 = *(unsigned int *)(a3 + 8);
  }
  *(_BYTE *)(*(_QWORD *)a3 + v10) = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v11 = (unsigned int)v41;
      v12 = 0;
      if ( (_DWORD)v41 )
      {
        do
        {
          while ( v40[v12] != 37 )
          {
            if ( v11 == ++v12 )
              goto LABEL_10;
          }
          *(_BYTE *)(*(_QWORD *)a3 + v12++) = a0123456789abcd_0[sub_16C6CF0() & 0xF];
        }
        while ( v11 != v12 );
      }
LABEL_10:
      v13 = *(_BYTE **)a3;
      if ( a6 == 1 )
        break;
      if ( a6 == 2 )
      {
        v44[0] = 257;
        if ( *v13 )
        {
          v43[0] = (__int64)v13;
          LOBYTE(v44[0]) = 3;
        }
        v14 = 0;
        v20 = sub_16C51A0((__int64)v43, 0);
        v22 = v21;
        v29 = v20;
        v25 = sub_2241E50(v43, 0, v20, v23, v24);
        v15 = v29;
        if ( v22 == v25 && (_DWORD)v29 == 2 )
          goto LABEL_36;
        if ( (_DWORD)v29 )
        {
          v18 = v29;
          goto LABEL_18;
        }
      }
      else
      {
        v44[0] = 257;
        if ( *v13 )
        {
          v43[0] = (__int64)v13;
          LOBYTE(v44[0]) = 3;
        }
        v14 = 0;
        v16 = sub_16C4FF0((__int64)v43, 0, 0x1F8u);
        v28 = v16;
        if ( !(_DWORD)v16 )
          goto LABEL_36;
        if ( v15 != sub_2241E50(v43, 0, v15, v16, v17) || v28 != 17 )
        {
          v18 = v28;
          goto LABEL_18;
        }
      }
    }
    v44[0] = 257;
    if ( *v13 )
    {
      v43[0] = (__int64)v13;
      LOBYTE(v44[0]) = 3;
    }
    v14 = a2;
    v16 = sub_16C5A80((__int64)v43, a2, 1, 3, a7, a5);
    v26 = v16;
    v27 = v16;
    if ( !(_DWORD)v16 )
      break;
    if ( v15 != sub_2241E50(v43, a2, v15, v16, v17) || v27 != 17 )
    {
      v18 = v26;
      goto LABEL_18;
    }
  }
LABEL_36:
  v18 = 0;
  sub_2241E40(v43, v14, v15, v16, v17);
LABEL_18:
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  return v18;
}
