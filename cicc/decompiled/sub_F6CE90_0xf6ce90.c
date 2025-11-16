// Function: sub_F6CE90
// Address: 0xf6ce90
//
__int64 __fastcall sub_F6CE90(int a1, __int64 *a2, __int64 a3, int a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r9
  __int64 *v8; // r15
  unsigned __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r8
  _QWORD *v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // edx
  char v20; // al
  unsigned int v21; // r9d
  _BYTE *v22; // rdi
  __int64 v24; // [rsp+8h] [rbp-158h]
  __int64 v25; // [rsp+10h] [rbp-150h]
  unsigned __int8 v27; // [rsp+18h] [rbp-148h]
  unsigned __int8 v28; // [rsp+18h] [rbp-148h]
  __int64 v29; // [rsp+20h] [rbp-140h] BYREF
  int v30; // [rsp+28h] [rbp-138h]
  _QWORD v31[2]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v32; // [rsp+40h] [rbp-120h] BYREF
  char *v33; // [rsp+48h] [rbp-118h]
  __int64 v34; // [rsp+50h] [rbp-110h]
  int v35; // [rsp+58h] [rbp-108h]
  char v36; // [rsp+5Ch] [rbp-104h]
  char v37; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v38; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+A8h] [rbp-B8h]
  _BYTE v40[176]; // [rsp+B0h] [rbp-B0h] BYREF

  if ( !a6 )
    return 1;
  v7 = (__int64)&a2[a3];
  v32 = 0;
  v8 = a2;
  v38 = v40;
  v39 = 0x800000000LL;
  v33 = &v37;
  v34 = 8;
  v35 = 0;
  v36 = 1;
  v29 = 0;
  v30 = 0;
  if ( a2 == (__int64 *)v7 )
  {
    v21 = 0;
    goto LABEL_15;
  }
  v11 = 8;
  v12 = 0;
  while ( 1 )
  {
    v13 = v12;
    v14 = *v8;
    if ( v12 >= v11 )
    {
      if ( v11 < (unsigned __int64)v12 + 1 )
      {
        a2 = (__int64 *)v40;
        v24 = v7;
        v25 = *v8;
        sub_C8D5F0((__int64)&v38, v40, v12 + 1LL, 0x10u, v14, v7);
        v13 = (unsigned int)v39;
        v7 = v24;
        v14 = v25;
      }
      v15 = &v38[16 * v13];
    }
    else
    {
      v15 = &v38[16 * v12];
      if ( !v15 )
        goto LABEL_8;
    }
    *v15 = -1;
    v15[1] = v14;
    v12 = v39;
LABEL_8:
    ++v12;
    ++v8;
    LODWORD(v39) = v12;
    if ( (__int64 *)v7 == v8 )
      break;
    v11 = HIDWORD(v39);
  }
  if ( v12 )
  {
    do
    {
      a2 = v31;
      v16 = (__int64 *)&v38[16 * v12 - 16];
      v17 = v16[1];
      v18 = *v16;
      v31[1] = v17;
      v31[0] = v18;
      LODWORD(v39) = v12 - 1;
      v19 = sub_F7E3C0(a1, (unsigned int)v31, a4, a7, (unsigned int)&v29, a5, a6, (__int64)&v32, (__int64)&v38);
      if ( (_BYTE)v19 )
        break;
      v12 = v39;
    }
    while ( (_DWORD)v39 );
    v20 = v36;
    v21 = v19;
  }
  else
  {
    v20 = v36;
    v21 = 0;
  }
  if ( v20 )
  {
LABEL_15:
    v22 = v38;
    if ( v38 != v40 )
      goto LABEL_16;
  }
  else
  {
    v28 = v21;
    _libc_free(v33, a2);
    v21 = v28;
    v22 = v38;
    if ( v38 != v40 )
    {
LABEL_16:
      v27 = v21;
      _libc_free(v22, a2);
      return v27;
    }
  }
  return v21;
}
