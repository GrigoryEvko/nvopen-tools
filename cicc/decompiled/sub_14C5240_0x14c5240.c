// Function: sub_14C5240
// Address: 0x14c5240
//
__int64 __fastcall sub_14C5240(__int64 *a1, const void *a2, __int64 a3)
{
  size_t v4; // r13
  __int64 v5; // rbx
  char *v6; // r12
  unsigned int v7; // ebx
  __int64 v8; // r13
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r11
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rdx
  __int64 v23; // rbx
  _BYTE *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r13
  char *v28; // rdi
  __int64 v29; // [rsp+18h] [rbp-118h]
  __int64 v30; // [rsp+20h] [rbp-110h]
  int v31; // [rsp+28h] [rbp-108h]
  __int64 v32; // [rsp+30h] [rbp-100h]
  unsigned int v33; // [rsp+38h] [rbp-F8h]
  unsigned int v34; // [rsp+3Ch] [rbp-F4h]
  _BYTE v35[16]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v36; // [rsp+50h] [rbp-E0h]
  void *dest; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+68h] [rbp-C8h]
  _BYTE v39[64]; // [rsp+70h] [rbp-C0h] BYREF
  void *src; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-78h]
  _BYTE v42[112]; // [rsp+C0h] [rbp-70h] BYREF

  v4 = 8 * a3;
  v5 = (8 * a3) >> 3;
  dest = v39;
  v33 = a3;
  v38 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_16CD150(&dest, v39, (8 * a3) >> 3, 8);
    v28 = (char *)dest + 8 * (unsigned int)v38;
    goto LABEL_38;
  }
  v6 = v39;
  if ( v4 )
  {
    v28 = v39;
LABEL_38:
    memcpy(v28, a2, v4);
    LODWORD(v4) = v38;
    v6 = (char *)dest;
  }
  LODWORD(v38) = v4 + v5;
  do
  {
    src = v42;
    v41 = 0x800000000LL;
    v34 = v33 - 1;
    if ( v33 == 1 )
    {
      v20 = 0;
LABEL_34:
      *((_QWORD *)src + v20) = *(_QWORD *)v6;
      v33 = v41 + 1;
      LODWORD(v41) = v41 + 1;
      goto LABEL_16;
    }
    v7 = 0;
    while ( 1 )
    {
      v8 = *(_QWORD *)&v6[8 * v7];
      v9 = *(__int64 **)&v6[8 * v7 + 8];
      v10 = *(_QWORD *)v8;
      v11 = *v9;
      if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 )
        v10 = 0;
      if ( *(_BYTE *)(v11 + 8) != 16 )
        BUG();
      v12 = *(_QWORD *)(v10 + 32);
      v13 = *(_QWORD *)(v11 + 32);
      if ( (unsigned int)v12 > (unsigned int)v13 )
      {
        v30 = *(_QWORD *)(v10 + 32);
        v31 = *(_QWORD *)(v11 + 32);
        v29 = *v9;
        v14 = sub_14C4F60((__int64)a1, 0, v13, (int)v12 - (int)v13);
        v36 = 257;
        v32 = v14;
        v15 = sub_1599EF0(v29);
        v16 = sub_14C50F0(a1, (__int64)v9, v15, v32, (__int64)v35);
        LODWORD(v12) = v30;
        LODWORD(v13) = v31;
        v9 = (__int64 *)v16;
      }
      v17 = sub_14C4F60((__int64)a1, 0, (int)v13 + (int)v12, 0);
      v36 = 257;
      v18 = sub_14C50F0(a1, v8, (__int64)v9, v17, (__int64)v35);
      v19 = (unsigned int)v41;
      if ( (unsigned int)v41 >= HIDWORD(v41) )
      {
        sub_16CD150(&src, v42, 0, 8);
        v19 = (unsigned int)v41;
      }
      v7 += 2;
      *((_QWORD *)src + v19) = v18;
      v20 = (unsigned int)(v41 + 1);
      LODWORD(v41) = v41 + 1;
      if ( v7 >= v34 )
        break;
      v6 = (char *)dest;
    }
    v21 = v33;
    v33 = v20;
    if ( (v21 & 1) != 0 )
    {
      v6 = (char *)dest + 8 * v34;
      if ( (unsigned int)v20 >= HIDWORD(v41) )
      {
        sub_16CD150(&src, v42, 0, 8);
        v20 = (unsigned int)v41;
      }
      goto LABEL_34;
    }
LABEL_16:
    v22 = v33;
    if ( v33 <= (unsigned __int64)(unsigned int)v38 )
    {
      if ( v33 )
        memmove(dest, src, 8LL * v33);
      v24 = src;
      LODWORD(v38) = v33;
    }
    else
    {
      if ( v33 > (unsigned __int64)HIDWORD(v38) )
      {
        v23 = 0;
        LODWORD(v38) = 0;
        sub_16CD150(&dest, v39, v33, 8);
        v22 = (unsigned int)v41;
      }
      else
      {
        v23 = 8LL * (unsigned int)v38;
        if ( (_DWORD)v38 )
        {
          memmove(dest, src, 8LL * (unsigned int)v38);
          v22 = (unsigned int)v41;
        }
      }
      v24 = src;
      v25 = 8 * v22;
      if ( (char *)src + v23 != (char *)src + v25 )
      {
        memcpy((char *)dest + v23, (char *)src + v23, v25 - v23);
        v24 = src;
      }
      LODWORD(v38) = v33;
    }
    if ( v24 != v42 )
      _libc_free((unsigned __int64)v24);
    v6 = (char *)dest;
  }
  while ( v33 > 1 );
  v26 = *(_QWORD *)dest;
  if ( dest != v39 )
    _libc_free((unsigned __int64)dest);
  return v26;
}
