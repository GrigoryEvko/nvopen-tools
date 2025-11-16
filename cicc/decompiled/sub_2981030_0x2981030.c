// Function: sub_2981030
// Address: 0x2981030
//
void __fastcall sub_2981030(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // r10
  __int64 *v16; // rax
  __int64 v17; // r15
  _BYTE *v18; // rcx
  _BYTE *v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-68h]
  _BYTE *v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  unsigned __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-48h]
  unsigned __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-38h]

  v8 = sub_ACD640(*((_QWORD *)a2 + 1), 1, 0);
  sub_2980F90(a1, a3, v8, a2, a4, a5);
  v9 = *a2;
  if ( (unsigned __int8)v9 <= 0x1Cu )
  {
    if ( (_BYTE)v9 != 5 )
      return;
    v12 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (v12 & 0xFFF7) != 0x11 || (_WORD)v12 != 17 || (a2[1] & 4) == 0 )
      goto LABEL_31;
  }
  else
  {
    if ( (unsigned __int8)v9 > 0x36u )
      return;
    v10 = 0x40540000000000LL;
    if ( !_bittest64(&v10, v9) || (_BYTE)v9 != 46 || (a2[1] & 4) == 0 )
    {
LABEL_6:
      v11 = 0x40540000000000LL;
      v12 = (unsigned __int8)v9 - 29;
      if ( !_bittest64(&v11, v9) )
        return;
      goto LABEL_7;
    }
  }
  v18 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( v18 )
  {
    v19 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v19 == 17 )
    {
      sub_2980F90(a1, a3, (__int64)v19, v18, a4, a5);
      return;
    }
  }
  if ( (unsigned __int8)v9 > 0x1Cu )
    goto LABEL_6;
  v12 = *((unsigned __int16 *)a2 + 1);
LABEL_31:
  if ( (v12 & 0xFFFD) != 0xD && (v12 & 0xFFF7) != 0x11 )
    return;
LABEL_7:
  if ( v12 != 25 )
    return;
  if ( (a2[1] & 4) == 0 )
    return;
  v21 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( !v21 )
    return;
  v13 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v13 != 17 )
    return;
  v14 = *(_DWORD *)(v13 + 32);
  v24 = v14;
  if ( v14 <= 0x40 )
  {
    v23 = 1;
    v15 = v13 + 24;
    v26 = v14;
LABEL_13:
    v25 = v23;
    goto LABEL_14;
  }
  v20 = v13;
  sub_C43690((__int64)&v23, 1, 0);
  v13 = v20;
  v26 = v24;
  v15 = v20 + 24;
  if ( v24 <= 0x40 )
    goto LABEL_13;
  sub_C43780((__int64)&v25, (const void **)&v23);
  v13 = v20;
  v15 = v20 + 24;
LABEL_14:
  v22 = v13;
  sub_C47AC0((__int64)&v25, v15);
  v16 = (__int64 *)sub_BD5C60(v22);
  v17 = sub_ACCFD0(v16, (__int64)&v25);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  sub_2980F90(a1, a3, v17, v21, a4, a5);
  if ( v24 > 0x40 )
  {
    if ( v23 )
      j_j___libc_free_0_0(v23);
  }
}
