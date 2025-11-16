// Function: sub_34A41A0
// Address: 0x34a41a0
//
__int64 __fastcall sub_34A41A0(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  char *v18; // rdi
  unsigned int v19; // esi
  __int64 v20; // [rsp+0h] [rbp-1F0h] BYREF
  char *v21; // [rsp+8h] [rbp-1E8h] BYREF
  int v22; // [rsp+10h] [rbp-1E0h]
  char v23; // [rsp+18h] [rbp-1D8h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-198h]
  __int128 v25; // [rsp+60h] [rbp-190h]
  __int64 v26; // [rsp+70h] [rbp-180h]
  _BYTE *v27; // [rsp+78h] [rbp-178h] BYREF
  __int64 v28; // [rsp+80h] [rbp-170h]
  _BYTE v29[64]; // [rsp+88h] [rbp-168h] BYREF
  int v30; // [rsp+C8h] [rbp-128h]
  unsigned __int64 v31; // [rsp+D0h] [rbp-120h]
  unsigned __int64 v32; // [rsp+D8h] [rbp-118h]
  __int64 v33; // [rsp+E0h] [rbp-110h]
  char *v34; // [rsp+E8h] [rbp-108h] BYREF
  __int64 v35; // [rsp+F0h] [rbp-100h]
  _BYTE v36[64]; // [rsp+F8h] [rbp-F8h] BYREF
  unsigned int v37; // [rsp+138h] [rbp-B8h]
  __int128 v38; // [rsp+140h] [rbp-B0h]
  __int64 v39; // [rsp+150h] [rbp-A0h]
  char *v40; // [rsp+158h] [rbp-98h] BYREF
  __int64 v41; // [rsp+160h] [rbp-90h]
  _BYTE v42[64]; // [rsp+168h] [rbp-88h] BYREF
  int v43; // [rsp+1A8h] [rbp-48h]
  unsigned __int64 v44; // [rsp+1B0h] [rbp-40h]
  unsigned __int64 v45; // [rsp+1B8h] [rbp-38h]

  sub_34A3D10((__int64)&v20, a2, a3, a4, a5, a6);
  v10 = v24;
  v39 = 0;
  v41 = 0x400000000LL;
  v11 = v25;
  v40 = v42;
  v44 = 0;
  v45 = 0;
  v43 = -1;
  if ( v24 == -1 && v25 == 0 || a4 <= (unsigned __int64)v25 + v24 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x400000000LL;
    *(_DWORD *)(a1 + 88) = -1;
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 112) = 0;
    *(_QWORD *)(a1 + 120) = a1 + 136;
    *(_QWORD *)(a1 + 128) = 0x400000000LL;
    *(_DWORD *)(a1 + 200) = -1;
    *(_QWORD *)(a1 + 208) = 0;
    *(_QWORD *)(a1 + 216) = 0;
  }
  else
  {
    v13 = v20;
    v27 = v29;
    v26 = v20;
    v28 = 0x400000000LL;
    if ( v22 )
    {
      sub_349DB40((__int64)&v27, (__int64)&v21, v24, v20, v8, v9);
      v10 = v24;
      v11 = v25;
    }
    v31 = v11;
    v30 = v10;
    v32 = *((_QWORD *)&v25 + 1);
    if ( (_DWORD)v10 != -1 )
    {
      if ( a4 <= *((_QWORD *)&v25 + 1) )
      {
LABEL_15:
        if ( a4 >= v31 )
          v30 = a4 - v31;
      }
      else
      {
        while ( 1 )
        {
          v14 = (__int64)&v27[16 * (unsigned int)v28 - 16];
          v10 = (unsigned int)(*(_DWORD *)(v14 + 12) + 1);
          *(_DWORD *)(v14 + 12) = v10;
          v13 = (unsigned int)v28;
          if ( (_DWORD)v10 == *(_DWORD *)&v27[16 * (unsigned int)v28 - 8] )
          {
            v19 = *(_DWORD *)(v26 + 192);
            if ( v19 )
            {
              sub_F03D40((__int64 *)&v27, v19);
              v13 = (unsigned int)v28;
            }
          }
          if ( !(_DWORD)v13 || *((_DWORD *)v27 + 3) >= *((_DWORD *)v27 + 2) )
            break;
          v30 = 0;
          v13 *= 16;
          v15 = (__int64)&v27[v13 - 16];
          v31 = *(_QWORD *)(*(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 12));
          v10 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 12);
          v32 = *(_QWORD *)(v10 + 8);
          if ( a4 <= v32 )
            goto LABEL_15;
        }
        v30 = -1;
        v31 = 0;
        v32 = 0;
      }
    }
    v34 = v36;
    v33 = v20;
    v35 = 0x400000000LL;
    if ( v22 )
      sub_349DB40((__int64)&v34, (__int64)&v21, v10, v13, v8, v9);
    v16 = (unsigned int)v28;
    v40 = v42;
    v37 = v24;
    v38 = v25;
    v39 = v26;
    v41 = 0x400000000LL;
    if ( (_DWORD)v28 )
      sub_349DB40((__int64)&v40, (__int64)&v27, v10, (unsigned int)v28, v8, v9);
    v17 = (unsigned int)v35;
    v43 = v30;
    v44 = v31;
    v45 = v32;
    *(_QWORD *)a1 = v33;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x400000000LL;
    if ( (_DWORD)v17 )
      sub_349DC20(a1 + 8, &v34, v17, v16, v8, v9);
    *(_DWORD *)(a1 + 88) = v37;
    *(_OWORD *)(a1 + 96) = v38;
    *(_QWORD *)(a1 + 112) = v39;
    *(_QWORD *)(a1 + 120) = a1 + 136;
    *(_QWORD *)(a1 + 128) = 0x400000000LL;
    if ( (_DWORD)v41 )
      sub_349DC20(a1 + 120, &v40, v17, v16, v8, v9);
    v18 = v40;
    *(_DWORD *)(a1 + 200) = v43;
    *(_QWORD *)(a1 + 208) = v44;
    *(_QWORD *)(a1 + 216) = v45;
    if ( v18 != v42 )
      _libc_free((unsigned __int64)v18);
    if ( v34 != v36 )
      _libc_free((unsigned __int64)v34);
    if ( v27 != v29 )
      _libc_free((unsigned __int64)v27);
  }
  if ( v21 != &v23 )
    _libc_free((unsigned __int64)v21);
  return a1;
}
